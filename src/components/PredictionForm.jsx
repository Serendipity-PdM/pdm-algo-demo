import React, { useState } from 'react';

const PredictionForm = () => {
  const [predictedRUL, setPredictedRUL] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = () => {
    setIsLoading(true);
    setTimeout(() => {
      const mockRUL = Math.floor(Math.random() * 150) + 1;
      setPredictedRUL(mockRUL);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <section className="py-12 px-6 max-w-xl mx-auto text-center">
      <h3 className="text-2xl font-semibold mb-4 text-gray-800">Predict RUL</h3>

      <p className="text-gray-600 mb-6">
        Click the button below to simulate a Remaining Useful Life (RUL) prediction.
        We'll connect this to our real ML model later.
      </p>

      <button
        onClick={handlePredict}
        className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded transition"
        disabled={isLoading}
      >
        {isLoading ? 'Predicting...' : 'Predict RUL'}
      </button>

      {predictedRUL !== null && (
        <div className="mt-8 p-4 bg-green-100 border border-green-300 rounded text-green-800 text-lg font-bold shadow-sm">
          📈 Estimated RUL: {predictedRUL} cycles
        </div>
      )}
    </section>
  );
};

export default PredictionForm;
