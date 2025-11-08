import React from 'react';

export const Card = ({ label, value, sub, positive, negative, children }) => {
  return (
    <div className="card">
      {label && <div className="card-label">{label}</div>}
      {value && (
        <div className={`card-value ${positive ? 'pos' : ''} ${negative ? 'neg' : ''}`}>
          {value}
        </div>
      )}
      {sub && <div className="card-sub">{sub}</div>}
      {children}
    </div>
  );
};

export const Button = ({ children, variant = 'default', onClick, className = '' }) => {
  const variantClass = variant === 'primary' ? 'btn-primary' : 
                       variant === 'secondary' ? 'btn-secondary' : 
                       'btn';
  
  return (
    <button className={`${variantClass} ${className}`} onClick={onClick}>
      {children}
    </button>
  );
};

export const Switch = ({ checked, onChange, label }) => {
  return (
    <div className="toggle-group">
      {label && <span>{label}</span>}
      <label className="switch">
        <input type="checkbox" checked={checked} onChange={onChange} />
        <span className="slider"></span>
      </label>
    </div>
  );
};

export const InputGroup = ({ children, className = '' }) => {
  return <div className={`input-group ${className}`}>{children}</div>;
};

export const ParamItem = ({ label, description, children }) => {
  return (
    <div className="param-item">
      <div className="param-label">{label}</div>
      {children}
      {description && <div className="param-desc">{description}</div>}
    </div>
  );
};
