import React from 'react';

const Header = () => {
  return (
    <header className="w-full bg-gray-900 text-white px-6 py-4 flex items-center justify-between shadow">
      <h1 className="text-xl font-bold">
        Serendipity PM – FD001 Demo
      </h1>
      <a
        href="https://github.com/Serendipity-PdM"
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm hover:underline hover:text-gray-300 transition"
      >
        GitHub ↗
      </a>
    </header>
  );
};

export default Header;
