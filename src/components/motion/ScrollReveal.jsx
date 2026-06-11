import React, { useEffect, useRef, useState } from 'react';

export default function ScrollReveal({ children, delay = 0, className = '' }) {
    const [isVisible, setIsVisible] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const currentRef = ref.current;
        if (!currentRef) return;

        // Use IntersectionObserver as requested for performance
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.unobserve(entry.target);
                }
            },
            {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            }
        );

        observer.observe(currentRef);

        return () => {
            if (currentRef) observer.unobserve(currentRef);
        };
    }, []);

    return (
        <div
            ref={ref}
            className={`transition-all duration-[420ms] ease-[cubic-bezier(0.22,1,0.36,1)] will-change-transform ${className}`}
            style={{
                opacity: isVisible ? 1 : 0,
                transform: isVisible ? 'translate3d(0, 0, 0)' : 'translate3d(0, 20px, 0)',
                transitionDelay: `${delay}ms`
            }}
        >
            {children}
        </div>
    );
}
