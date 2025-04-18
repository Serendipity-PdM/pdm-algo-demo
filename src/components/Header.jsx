import React from 'react';

const Header = () => {
  return (
    <header className="w-full bg-gray-900 text-white p-4 flex items-center justify-between shadow-md">
      <h1 className="text-xl font-bold">
        Predictive Maintenance - FD001 Demo
      </h1>
      <a
        href="https://github.com/your-org/your-repo"
        target="_blank"
        rel="noopener noreferrer"
        className="hover:underline text-sm"
      >
        GitHub
      </a>
    </header>
  );
};

export default Header;
