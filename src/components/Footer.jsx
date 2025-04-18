import React from 'react';

const Footer = () => {
  return (
    <footer style={{ padding: '1rem', textAlign: 'center', marginTop: '3rem', fontSize: '0.9rem', color: '#888' }}>
      © {new Date().getFullYear()} PDM FD001 Demo — Built for academic purposes
    </footer>
  );
};

export default Footer;
