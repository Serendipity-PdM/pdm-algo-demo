import React from 'react';

const Hero = () => {
  return (
    <section className="bg-gray-100 py-12 px-6 text-center">
      <h2 className="text-3xl font-bold mb-4 text-gray-800">
        Predictive Maintenance Demo
      </h2>
      <p className="text-gray-700 max-w-xl mx-auto text-lg">
        This simple interface lets us visualize and test our machine learning model that predicts the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS FD001 dataset.
      </p>
    </section>
  );
};

export default Hero;
