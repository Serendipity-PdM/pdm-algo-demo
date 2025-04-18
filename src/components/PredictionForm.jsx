import React, { useState } from 'react';

const PredictionForm = () => {
  const [fileContent, setFileContent] = useState('');
  const [unit, setUnit] = useState('');
  const [predictedRUL, setPredictedRUL] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const parseTestFile = (raw) => {
    const lines = raw.trim().split('\n');
    const parsed = lines.map((line) => {
      return line.trim().split(/\s+/).slice(0, 26).map(Number);
    });

    return parsed.map((row) => ({
      unit: row[0],
      time: row[1],
      features: row.slice(2), // operational settings + 21 sensors = 24 features
    }));
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setFileContent(event.target.result);
    };
    reader.readAsText(file);
  };

  const handlePredict = async () => {
    if (!fileContent || !unit) {
      alert('Please upload the file and select a unit.');
      return;
    }

    const data = parseTestFile(fileContent);
    const selected = data.filter((row) => row.unit === parseInt(unit));
    if (selected.length < 25) {
      alert('This unit has fewer than 25 cycles.');
      return;
    }

    const last25 = selected.slice(-25);
    const sequence = last25.map((row) => {
      const baseFeatures = row.features; // 24 features
      const padded = [...baseFeatures, 0, 0, 0, 0, 0]; // Add 5 dummy slope features
      return padded;
    }); // shape: [25][29]
    
    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      });

      if (!res.ok) throw new Error('Prediction failed.');

      const json = await res.json();
      setPredictedRUL(json.rul);
    } catch (err) {
      alert('Error predicting RUL: ' + err.message);
    }
    setIsLoading(false);
  };

  return (
    <section className="py-12 px-6 max-w-xl mx-auto text-center">
      <h3 className="text-2xl font-semibold mb-4 text-gray-800">Predict RUL</h3>

      <div className="mb-4">
        <input type="file" accept=".txt" onChange={handleFileUpload} />
      </div>

      <div className="mb-6">
        <input
          type="number"
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
        <div className="mt-8 p-4 bg-green-100 border border-green-300 rounded text-green-800 text-lg font-bold shadow-sm">
          📈 Predicted RUL: {predictedRUL} cycles
        </div>
      )}
    </section>
  );
};

export default PredictionForm;
