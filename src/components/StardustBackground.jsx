import React, { useEffect, useRef } from 'react';

const StardustBackground = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        window.addEventListener('resize', resize);
        resize();

        const particles = [];
        // Deep vibrant colors scattered on dark theme. Matches right side of screenshot 1 vibe.
        const colors = ['#3b82f6', '#8b5cf6', '#f97316', '#ffffff', '#14b8a6', '#ef4444', '#10b981'];

        class Particle {
            constructor() {
                this.reset(true);
            }

            reset(initial = false) {
                // Source particles from the far left
                this.x = initial ? Math.random() * canvas.width : -Math.random() * 200;

                // Spread vertically, mostly near center but sprawling outwards
                this.y = (canvas.height / 2) + ((Math.random() - 0.5) * canvas.height * 1.5);

                this.size = Math.random() * 1.8 + 0.5;

                // Horizontal speed pushes to the right
                this.speedX = Math.random() * 2.5 + 0.5;

                // Gentle vertical scattering
                const distFromCenter = this.y - (canvas.height / 2);
                this.speedY = (distFromCenter * 0.0005) + (Math.random() - 0.5) * 0.8;

                this.color = colors[Math.floor(Math.random() * colors.length)];
                this.opacity = Math.random() * 0.8 + 0.2;
                this.life = initial ? Math.random() * 500 : 0;
                this.maxLife = Math.random() * 400 + 300;

                this.angle = Math.random() * Math.PI * 2;
                this.angleSpeed = (Math.random() - 0.5) * 0.02;
            }

            update(scrollY) {
                this.x += this.speedX;
                this.y += this.speedY + Math.sin(this.angle) * 0.5;
                this.angle += this.angleSpeed;

                // Parallax effect
                const parallaxY = this.y - (scrollY * this.size * 0.15);

                this.life++;
                // Fade out
                const fade = 1 - (this.life / this.maxLife);

                if (this.x > canvas.width || this.life >= this.maxLife) {
                    this.reset();
                }

                return { x: this.x, y: parallaxY, opacity: this.opacity * fade };
            }

            draw(pos) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.globalAlpha = Math.max(0, pos.opacity);
                ctx.fill();

                if (this.size > 1.2) {
                    ctx.shadowBlur = this.size * 3;
                    ctx.shadowColor = this.color;
                } else {
                    ctx.shadowBlur = 0;
                }
            }
        }

        // Dense starfield to match screenshot
        const particleCount = Math.min(window.innerWidth / 2, 700);
        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        let scrollY = window.scrollY;
        let targetScrollY = scrollY;
        const handleScroll = () => { targetScrollY = window.scrollY; };
        window.addEventListener('scroll', handleScroll, { passive: true });

        const animate = () => {
            scrollY += (targetScrollY - scrollY) * 0.08;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                const pos = p.update(scrollY);
                p.draw(pos);
            });
            animationFrameId = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('scroll', handleScroll);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 bg-[#020202]"
            style={{ pointerEvents: 'none', zIndex: 0 }}
        />
    );
};

export default StardustBackground;
