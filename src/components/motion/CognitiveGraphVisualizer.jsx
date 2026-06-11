import React, { useEffect, useRef, useState } from 'react';

export default function CognitiveGraphVisualizer() {
    const [isVisible, setIsVisible] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const currentRef = ref.current;
        if (!currentRef) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                }
            },
            { threshold: 0.2 }
        );

        observer.observe(currentRef);

        return () => {
            if (currentRef) observer.unobserve(currentRef);
        };
    }, []);

    // AGI Evolutionary Framework - 7 Core Intelligence Layers
    // Fluid, non-rigid layout with distinct functional clusters representing AGI capabilities
    const nodes = [
        // Global Workspace Layer (Central Hub)
        { id: 0, cx: 200, cy: 200, r: 32, color: '#3b82f6', pulse: true, label: 'Global Workspace' },

        // Executive Controller Layer
        { id: 1, cx: 200, cy: 110, r: 22, color: '#8b5cf6', delay: 200 },

        // Perception Layer
        { id: 2, cx: 110, cy: 160, r: 18, color: '#0ea5e9', delay: 300 },

        // Action Layer
        { id: 3, cx: 290, cy: 160, r: 18, color: '#10b981', delay: 350 },

        // Working Memory Layer
        { id: 4, cx: 130, cy: 260, r: 16, color: '#f43f5e', delay: 400 },

        // Cognitive Graph Layer
        { id: 5, cx: 270, cy: 260, r: 20, color: '#f59e0b', delay: 450 },

        // Reasoning Layer
        { id: 6, cx: 200, cy: 290, r: 18, color: '#a855f7', delay: 500 },
    ];

    // Create intelligent clustering edges
    const edges = [];

    // Connect everything to the Global Workspace (Strongest links, broadcasting)
    for (let i = 1; i < nodes.length; i++) {
        edges.push({
            id: `e-core-${i}`, x1: nodes[0].cx, y1: nodes[0].cy, x2: nodes[i].cx, y2: nodes[i].cy,
            opacity: 0.3, width: 1.5, delay: nodes[i].delay, dist: Math.hypot(nodes[0].cx - nodes[i].cx, nodes[0].cy - nodes[i].cy)
        });
    }

    // Cross-layer connections (Cognitive Cycle)
    // Perceive -> Memory -> Graph -> Reason -> Executive -> Action
    const abstractEdges = [[2, 4], [4, 5], [5, 6], [6, 1], [1, 3], [3, 2], [1, 4], [6, 5]];
    abstractEdges.forEach(([i, j], idx) => {
        edges.push({
            id: `e-cross-${idx}`, x1: nodes[i].cx, y1: nodes[i].cy, x2: nodes[j].cx, y2: nodes[j].cy,
            opacity: 0.15, width: 0.8, delay: 600 + idx * 100, dist: Math.hypot(nodes[i].cx - nodes[j].cx, nodes[i].cy - nodes[j].cy)
        });
    });

    return (
        <div ref={ref} className="relative w-full h-full flex items-center justify-center p-4">
            {/* Background ambient glow matching the stardust theme */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(59,130,246,0.08)_0%,transparent_70%)] rounded-full blur-3xl pointer-events-none" />

            <svg viewBox="0 0 400 400" className="w-full max-w-[400px] h-auto overflow-visible drop-shadow-2xl">
                <defs>
                    <filter id="premium-glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="6" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    <filter id="core-glow" x="-100%" y="-100%" width="300%" height="300%">
                        <feGaussianBlur stdDeviation="12" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>

                    <linearGradient id="edge-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="rgba(255,255,255,0)" />
                        <stop offset="50%" stopColor="rgba(255,255,255,0.4)" />
                        <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                    </linearGradient>
                </defs>

                {/* Draw edges with premium sweeping opacity */}
                {edges.map((edge) => (
                    <g key={edge.id}>
                        <line
                            x1={edge.x1}
                            y1={edge.y1}
                            x2={edge.x2}
                            y2={edge.y2}
                            stroke="url(#edge-grad)"
                            strokeWidth={edge.width}
                            opacity={edge.opacity}
                            className={`transition-all duration-[1200ms] ease-[cubic-bezier(0.22,1,0.36,1)]`}
                            style={{
                                strokeDasharray: edge.dist,
                                strokeDashoffset: isVisible ? '0' : edge.dist,
                                transitionDelay: `${edge.delay}ms`
                            }}
                        />
                        {/* Elegant data pulse (not particles, but glowing energy bursts) */}
                        {isVisible && Math.random() > 0.3 && (
                            <circle r="1.5" fill="#fff" filter="url(#premium-glow)" opacity="0">
                                <animateMotion
                                    dur={`${edge.dist / 30}s`}
                                    repeatCount="indefinite"
                                    path={`M ${edge.x1} ${edge.y1} L ${edge.x2} ${edge.y2}`}
                                    begin={`${edge.delay / 1000 + Math.random()}s`}
                                    calcMode="spline"
                                    keySplines="0.4 0 0.2 1"
                                    keyTimes="0;1"
                                />
                                <animate
                                    attributeName="opacity"
                                    values="0;1;0"
                                    keyTimes="0;0.5;1"
                                    dur={`${edge.dist / 30}s`}
                                    repeatCount="indefinite"
                                    begin={`${edge.delay / 1000 + Math.random()}s`}
                                />
                            </circle>
                        )}
                    </g>
                ))}

                {/* Draw Premium Nodes */}
                {nodes.map((node) => (
                    <g
                        key={node.id}
                        className="transition-all duration-[900ms] ease-[cubic-bezier(0.22,1,0.36,1)]"
                        style={{
                            opacity: isVisible ? 1 : 0,
                            transform: isVisible ? 'scale(1) translateY(0)' : 'scale(0.8) translateY(10px)',
                            transformOrigin: `${node.cx}px ${node.cy}px`,
                            transitionDelay: `${node.delay || 0}ms`
                        }}
                    >
                        {/* Outer ambient blur */}
                        {node.pulse && (
                            <circle
                                cx={node.cx}
                                cy={node.cy}
                                r={node.r * 2.5}
                                fill={node.color}
                                opacity="0.1"
                                filter="url(#core-glow)"
                                className="animate-pulse-slow origin-center"
                            />
                        )}

                        {/* Main glassmorphic node body */}
                        <circle
                            cx={node.cx}
                            cy={node.cy}
                            r={node.r}
                            fill={node.color}
                            opacity="0.8"
                            filter={node.id === 0 ? "url(#core-glow)" : "url(#premium-glow)"}
                            className={`transition-transform duration-500 hover:scale-110 cursor-pointer ${node.id === 0 ? 'animate-float' : ''}`}
                            style={{ transformOrigin: `${node.cx}px ${node.cy}px` }}
                        />

                        {/* Specular highlight / Core crystalline structure */}
                        <circle
                            cx={node.cx - node.r * 0.2}
                            cy={node.cy - node.r * 0.2}
                            r={node.id === 0 ? node.r * 0.4 : node.r * 0.3}
                            fill="#fff"
                            opacity="0.6"
                            filter="url(#premium-glow)"
                            className="pointer-events-none"
                        />

                        {/* AGI Label on Core */}
                        {node.id === 0 && (
                            <text
                                x={node.cx}
                                y={node.cy + node.r + 20}
                                fill="#fff"
                                fontSize="11"
                                fontWeight="600"
                                letterSpacing="0.1em"
                                textAnchor="middle"
                                opacity="0.7"
                                className="font-sans uppercase tracking-widest pointer-events-none"
                            >
                                AGI Core
                            </text>
                        )}
                    </g>
                ))}
            </svg>
        </div>
    );
}
