import React from 'react';
import { useNavigate } from 'react-router-dom';

const Header = () => {
  const navigate = useNavigate();

  return (
    <header className="w-full bg-gray-900 text-white px-6 py-4 flex items-center justify-between shadow">
      <h1 className="text-xl font-bold">
        Serendipity PM – FD001 Demo
      </h1>

      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate("/")}
          className="bg-green-500 hover:bg-green-600 text-white px-3 py-1.5 rounded text-sm transition"
        >
          Machine PM Prediction
        </button>
        <button
          onClick={() => navigate("/shift-risk-dashboard")}
          className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1.5 rounded text-sm transition"
        >
          Shift Risk Dashboard
        </button>
        <a
          href="https://github.com/Serendipity-PdM"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm hover:underline hover:text-gray-300 transition"
        >
          GitHub ↗
        </a>
      </div>
    </header>
  );
};

export default Header;
