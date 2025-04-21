import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  return (
    <footer className="mt-auto py-6 text-center text-sm text-gray-500 border-t border-gray-200">
      © {currentYear} PDM FD001 Demo — Built for academic use.
    </footer>
  );
};

export default Footer;
