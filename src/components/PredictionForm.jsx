import React, { useState } from 'react';

const PredictionForm = () => {
  const [fileContent, setFileContent] = useState('');
  const [rulFileContent, setRulFileContent] = useState('');
  const [unit, setUnit] = useState('');
  const [predictedRUL, setPredictedRUL] = useState(null);
  const [trueRUL, setTrueRUL] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const parseTestFile = (raw) => {
    const lines = raw.trim().split('\n');
    return lines.map((line) => {
      const cols = line.trim().split(/\s+/).slice(0, 26).map(Number);
      return { unit: cols[0], time: cols[1], features: cols.slice(2) };
    });
  };

  const parseRULFile = (raw) => {
    // Each line corresponds to the true RUL for units 1..N
    const lines = raw.trim().split('\n');
    const values = lines.map((line) => Number(line.trim().split(/\s+/)[0]));
    return values.reduce((acc, rul, idx) => {
      acc[idx + 1] = rul;
      return acc;
    }, {});
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => setFileContent(event.target.result);
    reader.readAsText(file);
  };

  const handleRULUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => setRulFileContent(event.target.result);
    reader.readAsText(file);
  };

  const handlePredict = async () => {
    if (!fileContent || !unit || !rulFileContent) {
      alert('Please upload test data, RUL file, and select a unit.');
      return;
    }

    const data = parseTestFile(fileContent);
    const selected = data.filter((row) => row.unit === Number(unit));
    if (selected.length === 0) {
      alert(`No data found for unit ${unit}.`);
      return;
    }

    // Take the last 25 cycles (or all if fewer than 25)
    const lastCycles = selected.length > 25 ? selected.slice(-25) : selected;
    const sequence = lastCycles.map((row) => row.features);

    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      });
      if (!res.ok) throw new Error('Prediction failed.');
      const { rul } = await res.json();
      setPredictedRUL(rul);

      // Use provided RUL file directly (true remaining cycles)
      const rulMap = parseRULFile(rulFileContent);
      setTrueRUL(rulMap[Number(unit)]);
    } catch (err) {
      alert('Error: ' + err.message);
    }
    setIsLoading(false);
  };

  return (
    <section className="py-12 px-6 max-w-xl mx-auto text-center">
      <h3 className="text-2xl font-semibold mb-4 text-gray-800">Predict RUL</h3>

      <div className="mb-4">
        <label className="block mb-2">Upload test data:</label>
        <input type="file" accept=".txt" onChange={handleFileUpload} />
      </div>
      <div className="mb-4">
        <label className="block mb-2">Upload RUL file:</label>
        <input type="file" accept=".txt" onChange={handleRULUpload} />
      </div>

      <div className="mb-4">
        <input
          type="number"
          min="0"
          placeholder="Enter unit number (e.g., 1)"
          value={unit}
          onChange={(e) => setUnit(e.target.value)}
          className="p-2 border rounded w-full max-w-xs"
        />
      </div>

      <button
        onClick={handlePredict}
        disabled={isLoading}
        className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded transition"
      >
        {isLoading ? 'Predicting...' : 'Predict RUL'}
      </button>

      {predictedRUL !== null && (
        <div className="mt-8 flex flex-col items-center space-y-4">
          <div className="flex space-x-4">
            <div className="p-4 bg-green-100 border border-green-300 rounded text-green-800 text-lg font-bold shadow-sm">
              📈 Predicted RUL: {Number(predictedRUL).toFixed(2)} cycles
            </div>
            {trueRUL !== null && (
              <div className="p-4 bg-blue-100 border border-blue-300 rounded text-blue-800 text-lg font-bold shadow-sm">
                🎯 True RUL: {trueRUL.toFixed(2)} cycles
              </div>
            )}
          </div>
          {trueRUL !== null && (
            <div
              className="p-4 bg-gray-100 border border-gray-300 rounded text-gray-800 text-lg font-bold shadow-sm"
              title="True RUL − Predicted RUL"
            >
              🔍 Difference: {Math.abs((trueRUL - predictedRUL).toFixed(2))} cycles
            </div>
          )}
        </div>
      )}
    </section>
  );
};

export default PredictionForm;
