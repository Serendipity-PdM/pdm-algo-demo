import React, { useState, useEffect } from 'react';

const PredictionForm = () => {
  const [fileContent, setFileContent] = useState('');
  const [rulFileContent, setRulFileContent] = useState('');
  const [unit, setUnit] = useState('');
  const [predictedRUL, setPredictedRUL] = useState(null);
  const [trueRUL, setTrueRUL] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [previewSequence, setPreviewSequence] = useState([]);

  const parseTestFile = (raw) => {
    const lines = raw.trim().split('\n');
    return lines.map((line) => {
      const cols = line.trim().split(/\s+/).slice(0, 26).map(Number);
      return { unit: cols[0], time: cols[1], features: cols.slice(2) };
    });
  };

  const parseRULFile = (raw) => {
    const lines = raw.trim().split('\n');
    const values = lines.map((line) => Number(line.trim().split(/\s+/)[0]));
    return values.reduce((acc, rul, idx) => {
      acc[idx + 1] = rul;
      return acc;
    }, {});
  };

  useEffect(() => {
    if (!fileContent || !unit) {
      setPreviewSequence([]);
      return;
    }
    const data = parseTestFile(fileContent);
    const selected = data.filter((row) => row.unit === Number(unit));
    if (!selected.length) {
      setPreviewSequence([]);
    } else {
      const lastCycles = selected.length > 25 ? selected.slice(-25) : selected;
      const seq = lastCycles.map((row) => row.features);
      setPreviewSequence(seq);
    }
  }, [fileContent, unit]);

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
    if (!previewSequence.length || !rulFileContent) {
      alert('Please upload test data, select a unit, and upload RUL file.');
      return;
    }

    setIsLoading(true);
    try {
      const payload = { sequence: previewSequence };
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Prediction failed.');
      const { rul } = await res.json();
      setPredictedRUL(rul);

      const rulMap = parseRULFile(rulFileContent);
      setTrueRUL(rulMap[Number(unit)]);
    } catch (err) {
      alert('Error: ' + err.message);
    }
    setIsLoading(false);
  };

  return (
    <section className="py-12 px-6 max-w-6xl mx-auto text-center">
      <h3 className="text-3xl font-bold mb-6 text-gray-800">Predict Machine RUL</h3>

      <div className="flex flex-col sm:flex-row sm:space-x-6 justify-center items-start sm:items-center mb-6">
        <div className="mb-4 sm:mb-0">
          <label className="block text-sm font-medium mb-2 text-gray-700">Upload test data:</label>
          <input
            type="file"
            accept=".txt"
            onChange={handleFileUpload}
            className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-white file:mr-4 file:py-2 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200 transition"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-700">Upload RUL file:</label>
          <input
            type="file"
            accept=".txt"
            onChange={handleRULUpload}
            className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-white file:mr-4 file:py-2 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-purple-100 file:text-purple-700 hover:file:bg-purple-200 transition"
          />
        </div>
      </div>

      <div className="mb-4">
        <input
          type="number"
          min="0"
          placeholder="Enter unit number (e.g., 1)"
          value={unit}
          onChange={(e) => setUnit(e.target.value)}
          className="p-2 border rounded w-full max-w-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      <button
        onClick={handlePredict}
        disabled={isLoading}
        className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded transition"
      >
        {isLoading ? 'Predicting...' : 'Predict RUL'}
      </button>
      <button
        onClick={() => {
          setUnit('');
          setPredictedRUL(null);
          setTrueRUL(null);
          setPreviewSequence([]);
        }}
        className="ml-4 mt-2 text-sm bg-red-100 hover:bg-red-200 text-red-700 font-semibold py-1 px-3 rounded border border-red-300 transition"
      >
        Clear
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

      {previewSequence.length > 0 && (
        <div className="mt-8 text-left overflow-x-auto max-w-full border rounded bg-white shadow">
          <h4 className="font-medium mb-2 text-gray-700 px-4 pt-4">Preview of last 25 cycles</h4>
          <table className="table-auto min-w-full border-collapse text-xs">
            <thead className="bg-gray-100">
              <tr>
                <th className="border border-gray-300 px-2 py-1 sticky left-0 bg-white">Cycle</th>
                {previewSequence[0].map((_, i) => (
                  <th key={i} className="border border-gray-300 px-2 py-1">f{i + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {previewSequence.map((row, idx) => (
                <tr key={idx}>
                  <td className="border border-gray-300 px-2 py-1 font-semibold text-center sticky left-0 bg-white">
                    {idx + 1}
                  </td>
                  {row.map((val, j) => (
                    <td key={j} className="border border-gray-300 px-2 py-1 text-right">
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default PredictionForm;
