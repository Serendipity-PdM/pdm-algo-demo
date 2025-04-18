import React from 'react';

const Hero = () => {
  return (
    <section style={{ padding: '2rem', textAlign: 'center', backgroundColor: '#f5f5f5' }}>
      <h2 style={{ fontSize: '2rem', marginBottom: '1rem' }}>
        Predictive Maintenance Demo
      </h2>
      <p style={{ maxWidth: '600px', margin: '0 auto', fontSize: '1.1rem' }}>
        This simple interface lets us visualize and test our machine learning model that predicts the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS FD001 dataset.
      </p>
    </section>
  );
};

export default Hero;
