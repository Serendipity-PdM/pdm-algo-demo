import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom"; 
import Header from "./components/Header.jsx";
import Hero from "./components/Hero.jsx";
import PredictionForm from "./components/PredictionForm.jsx";
import Footer from "./components/Footer.jsx";
import ShiftPredictionForm from "./components/ShiftPredictionForm.jsx";
import ShiftRiskDashboard from "./components/ShiftRiskDashboard.jsx";

const App = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <BrowserRouter>
        <Header />

        <Routes>
          <Route
            path="/"
            element={
              <>
                <Hero />
                <PredictionForm />
                <ShiftPredictionForm />
              </>
            }
          />
          <Route path="/shift-risk-dashboard" element={<ShiftRiskDashboard />} />
        </Routes>

        <div className="flex-grow" />
        <Footer />
      </BrowserRouter>
    </div>
  );
};

export default App;
