import React, { useEffect, useRef, useState, Children, isValidElement, cloneElement } from 'react';

// Recursive function to split text into words and letters while preserving HTML/React elements
function createSplitNodes(node, isVisible, baseDelay, globalIndexRef) {
    if (typeof node === 'string' || typeof node === 'number') {
        const text = String(node);
        const words = text.split(/(\s+)/); // Preserve spaces

        return words.map((word, wIdx) => {
            // If it's just whitespace, render as is
            if (/^\s+$/.test(word)) {
                return <span key={`space-${wIdx}`} className="whitespace-pre">{word}</span>;
            }

            // If it's a word, split into letters
            const letters = word.split('');
            return (
                <span key={`word-${wIdx}`} className="inline-block whitespace-nowrap">
                    {letters.map((letter, lIdx) => {
                        const currentIndex = globalIndexRef.current++;
                        return (
                            <span
                                key={`${wIdx}-${lIdx}`}
                                className="inline-block transition-all duration-[420ms] ease-[cubic-bezier(0.22,1,0.36,1)] will-change-transform"
                                style={{
                                    opacity: isVisible ? 1 : 0,
                                    transform: isVisible ? 'translate3d(0, 0, 0)' : 'translate3d(0, 15px, 0)',
                                    transitionDelay: `${baseDelay + currentIndex * 15}ms`,
                                }}
                            >
                                {letter}
                            </span>
                        );
                    })}
                </span>
            );
        });
    }

    // If it's a React element (like <span> or <br/>), traverse its children
    if (isValidElement(node)) {
        if (node.type === 'br') return node;

        const childNodes = Children.toArray(node.props.children);
        return cloneElement(node, {
            ...node.props,
            children: childNodes.map((child) =>
                createSplitNodes(child, isVisible, baseDelay, globalIndexRef)
            )
        });
    }

    return node;
}

export default function LetterReveal({ children, delay = 0, className = '' }) {
    const [isVisible, setIsVisible] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const currentRef = ref.current;
        if (!currentRef) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.unobserve(entry.target);
                }
            },
            { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
        );

        observer.observe(currentRef);

        return () => {
            if (currentRef) observer.unobserve(currentRef);
        };
    }, []);

    // Use a ref to track the global letter index across nested elements for continuous staggered delay
    const globalIndexRef = useRef(0);
    globalIndexRef.current = 0; // Reset on render

    const splitChildren = Children.toArray(children).map((child) =>
        createSplitNodes(child, isVisible, delay, globalIndexRef)
    );

    return (
        <div ref={ref} className={`inline-block ${className}`}>
            {splitChildren}
        </div>
    );
}
