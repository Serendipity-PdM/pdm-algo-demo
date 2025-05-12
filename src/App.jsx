import React from 'react';
import Header from './components/Header.jsx';
import Hero from './components/Hero.jsx';
import PredictionForm from './components/PredictionForm.jsx';
import Footer from './components/Footer.jsx';
import ShiftPredictionForm from "./components/ShiftPredictionForm";
import ShiftRiskDashboard from "./components/ShiftRiskDashboard";

const App = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <Hero />
      <PredictionForm />
      <ShiftPredictionForm />
      <ShiftRiskDashboard />
      <div className="flex-grow" /> {}
      <Footer />
    </div>
  );
};

export default App;
