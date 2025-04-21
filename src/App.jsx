import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import PredictionForm from './components/PredictionForm';
import Footer from './components/Footer';

const App = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <Hero />
      <PredictionForm />
      <div className="flex-grow" /> {}
      <Footer />
    </div>
  );
};

export default App;
