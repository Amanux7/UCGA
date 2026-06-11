import React, { useRef, useState } from 'react';

export default function MagneticCTA({ children, className = '', onClick }) {
    const ref = useRef(null);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isHovered, setIsHovered] = useState(false);

    const handleMouseMove = (e) => {
        if (!ref.current) return;
        const { clientX, clientY } = e;
        const { left, top, width, height } = ref.current.getBoundingClientRect();

        // Calculate distance from center
        const x = (clientX - (left + width / 2)) * 0.2;
        const y = (clientY - (top + height / 2)) * 0.2;

        // Max constraint 6px as requested for subtle magnetic pull
        setPosition({
            x: Math.min(Math.max(x, -6), 6),
            y: Math.min(Math.max(y, -6), 6)
        });
    };

    const handleMouseEnter = () => setIsHovered(true);

    const handleMouseLeave = () => {
        setIsHovered(false);
        setPosition({ x: 0, y: 0 });
    };

    return (
        <button
            ref={ref}
            onClick={onClick}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            className={`relative group inline-flex items-center justify-center transition-all duration-[280ms] ease-[cubic-bezier(0.22,1,0.36,1)] will-change-transform ${className}`}
            style={{
                transform: isHovered
                    ? `translate3d(${position.x}px, ${position.y - 3}px, 0) scale3d(1.02, 1.02, 1.02)`
                    : 'translate3d(0, 0, 0) scale3d(1, 1, 1)',
                boxShadow: isHovered
                    ? '0 15px 30px -5px rgba(0,0,0,0.6), 0 0 20px 0px rgba(255,255,255,0.08)'
                    : '0 4px 10px rgba(0,0,0,0.2)',
            }}
        >
            <div
                className="absolute inset-0 rounded-full transition-all duration-[280ms] ease-[cubic-bezier(0.22,1,0.36,1)] group-hover:bg-white/5"
                style={{ transform: 'translateZ(0)' }}
            />
            <span className="relative z-10 flex items-center gap-2">
                {children}
            </span>
        </button>
    );
}
