import React from 'react';

const Footer = () => {
  return (
    <footer className="mt-16 py-6 text-center text-sm text-gray-500 border-t border-gray-200">
      © {new Date().getFullYear()} PDM FD001 Demo — Built for academic use
    </footer>
  );
};

export default Footer;
