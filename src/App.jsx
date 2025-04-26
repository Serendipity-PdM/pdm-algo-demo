import React from 'react';
import Header from './components/Header.jsx';
import Hero from './components/Hero.jsx';
import PredictionForm from './components/PredictionForm.jsx';
import Footer from './components/Footer.jsx';

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
