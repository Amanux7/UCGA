import React, { useRef, useState } from 'react';

export default function TiltCard({ children, className = '' }) {
  const ref = useRef(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e) => {
    if (!ref.current) return;
    
    // Read synchronously for safety
    const rect = ref.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    window.requestAnimationFrame(() => {
        const xPct = mouseX / width - 0.5; // -0.5 to 0.5 range
        const yPct = mouseY / height - 0.5;
        
        // Exact 3 degrees max tilt as requested for intelligence minimal feel
        setRotation({
          x: -yPct * 6, // 6 * 0.5 = 3deg max
          y: xPct * 6,
        });
    });
  };

  const handleMouseEnter = () => setIsHovered(true);
  
  const handleMouseLeave = () => {
    setIsHovered(false);
    setRotation({ x: 0, y: 0 });
  };

  return (
    <div className={`perspective-[1000px] ${className}`}>
      <div
        ref={ref}
        onMouseMove={handleMouseMove}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="relative w-full h-full transition-all duration-[420ms] ease-[cubic-bezier(0.22,1,0.36,1)] will-change-transform"
        style={{
          transform: isHovered 
            ? `translate3d(0, -6px, 0) scale3d(1.02, 1.02, 1.02) rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)` 
            : 'translate3d(0, 0, 0) scale3d(1, 1, 1) rotateX(0deg) rotateY(0deg)',
          boxShadow: isHovered 
            ? '0 25px 40px -15px rgba(0,0,0,0.8), 0 0 20px -5px rgba(255,255,255,0.05)' 
            : '0 4px 10px rgba(0,0,0,0.2)',
          transformStyle: 'preserve-3d'
        }}
      >
        {children}
      </div>
    </div>
  );
}
