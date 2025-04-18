import React, { useState } from 'react';

const PredictionForm = () => {
  const [predictedRUL, setPredictedRUL] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // For now, this is just mock behavior
  const handlePredict = () => {
    setIsLoading(true);

    // Simulate prediction delay
    setTimeout(() => {
      const mockRUL = Math.floor(Math.random() * 150) + 1;
      setPredictedRUL(mockRUL);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <section style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
        Predict RUL
      </h3>

      <p style={{ marginBottom: '1rem', fontSize: '0.95rem', color: '#555' }}>
        Click the button below to simulate a Remaining Useful Life (RUL) prediction.
        Later, this will connect to our actual LSTM model output.
      </p>

      <button
        onClick={handlePredict}
        style={{
          padding: '0.5rem 1rem',
          fontSize: '1rem',
          backgroundColor: '#007bff',
          color: '#fff',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
        }}
      >
        {isLoading ? 'Predicting...' : 'Predict RUL'}
      </button>

      {predictedRUL !== null && (
        <div style={{ marginTop: '1.5rem', fontSize: '1.2rem', fontWeight: 'bold' }}>
          📈 Estimated RUL: {predictedRUL} cycles
        </div>
      )}
    </section>
  );
};

export default PredictionForm;
