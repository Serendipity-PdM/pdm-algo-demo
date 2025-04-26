import React from 'react';
import Header from './components/Header.tsx';
import Hero from './components/Hero.tsx';
import PredictionForm from './components/PredictionForm.tsx';
import Footer from './components/Footer.tsx';

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
