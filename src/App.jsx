import React from 'react';
import { motion } from 'framer-motion';
import {
  Network, Download, Brain, Code2,
  Terminal, Frame, Box, Command, Code, Sparkles, Folder, Fingerprint, ShieldCheck, Wand2
} from 'lucide-react';
import StardustBackground from './components/StardustBackground';
import MagneticCTA from './components/motion/MagneticCTA';
import TiltCard from './components/motion/TiltCard';
import ScrollReveal from './components/motion/ScrollReveal';
import LetterReveal from './components/motion/LetterReveal';
import CognitiveGraphVisualizer from './components/motion/CognitiveGraphVisualizer';
import InteractiveSimulator from './components/InteractiveSimulator';

const Nav = () => (
  <nav className="fixed top-0 w-full z-50 py-5 px-6 md:px-12 flex justify-between items-center bg-black/50 backdrop-blur-md border-b border-white/5">
    <div className="flex items-center gap-3">
      <img src="/ucga_logo.png" alt="UCGA Logo" className="h-10 md:h-12 w-auto object-contain" />
    </div>
    <div className="hidden md:flex items-center gap-8 text-[13px] font-medium text-gray-400">
      <a href="#" className="hover:text-white transition-colors">Product</a>
      <a href="#" className="flex items-center gap-1 hover:text-white transition-colors">Use Cases <span className="text-[9px] opacity-70">▼</span></a>
      <a href="#" className="hover:text-white transition-colors">Pricing</a>
      <a href="#" className="hover:text-white transition-colors">Blog</a>
      <a href="#" className="flex items-center gap-1 hover:text-white transition-colors">Resources <span className="text-[9px] opacity-70">▼</span></a>
    </div>
    <MagneticCTA className="px-5 py-2 rounded-full bg-white text-black text-[13px] font-semibold hover:bg-gray-200 transition-colors">
      Coming Soon
    </MagneticCTA>
  </nav>
);

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
      <div className="container mx-auto px-6 relative z-10 text-center flex flex-col items-center">
        <ScrollReveal className="flex flex-col items-center w-full">
          <div className="flex items-center gap-3 mb-10">
            <img src="/ucga_logo.png" alt="UCGA Logo" className="h-16 md:h-20 w-auto object-contain drop-shadow-xl" />
          </div>
          <h1 className="text-6xl md:text-8xl lg:text-[110px] font-semibold tracking-[-0.03em] leading-[1] mb-14 max-w-[1200px] text-white">
            <LetterReveal>
              Experience AGI with the<br />
              next-generation<br />
              Cognitive Architecture
            </LetterReveal>
          </h1>
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <MagneticCTA className="px-8 py-3.5 rounded-full bg-neutral-900 border border-white/10 text-white font-medium flex items-center gap-3 shadow-2xl hover:bg-neutral-800 transition-colors text-[15px]">
              Coming Soon
            </MagneticCTA>
            <MagneticCTA className="px-8 py-3.5 rounded-full border border-white/10 bg-white/5 text-white font-medium hover:bg-white/10 transition-colors backdrop-blur-md text-[15px]">
              Explore use cases
            </MagneticCTA>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

const FeatureOne = () => (
  <section className="py-32 container mx-auto px-6 md:px-12 grid lg:grid-cols-2 gap-16 items-center">
    <ScrollReveal delay={0} className="max-w-md">
      <h2 className="text-4xl md:text-[52px] font-medium tracking-tight mb-8 text-white leading-[1.1]">
        <LetterReveal>Recursive Cognitive Core</LetterReveal>
      </h2>
      <p className="text-gray-400 text-lg leading-relaxed">
        UCGA replaces static layers with a recursive refinement loop. It offers built-in self-evaluation, persistent attention-based memory, and dynamic error correction through a modular graph.
      </p>
    </ScrollReveal>
    <ScrollReveal delay={100} className="h-[500px]">
      <TiltCard className="w-full h-full">
        <div className="bg-[#050505] border border-white/10 rounded-3xl w-full h-full flex flex-col pt-6 relative overflow-hidden shadow-2xl">
          <div className="px-6 flex gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-red-500/80" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <div className="w-3 h-3 rounded-full bg-green-500/80" />
          </div>
          <div className="flex-1 border-t border-white/5 bg-[#0a0a0b] p-8 font-mono text-[13px] leading-8 overflow-hidden relative">
            <div className="text-gray-500 mb-4">// Implementation plan <span className="text-blue-400 text-xs ml-4 border border-blue-400/30 bg-blue-400/10 px-2 py-0.5 rounded">CognitiveNode.ts 1</span></div>
            <div><span className="text-blue-400">import</span> {'{'} NodeState {'}'} <span className="text-blue-400">from</span> <span className="text-green-400">'next/cognitive'</span>;</div>
            <br />
            <div><span className="text-purple-400">export default function</span> <span className="text-yellow-200">PerceptionLayer</span>() {'{'}</div>
            <div className="pl-4 text-gray-500 italic">  // UCGA Agent is writing...</div>
            <div className="pl-4 text-gray-300 overflow-hidden whitespace-nowrap border-r-2 border-white pr-2 animate-pulse">
              <span className="text-blue-400">return</span> {'<NodeState threshold={0.5} active={true} />'}
            </div>
            <div>{'}'}</div>
          </div>
          {/* Floating snippet */}
          <div className="absolute bottom-[-10px] right-8 bg-[#111] border border-white/10 rounded-xl p-4 shadow-2xl w-64 animate-float">
            <div className="flex items-center gap-2 mb-2">
              <Network className="w-4 h-4 text-orange-400" />
              <span className="text-[11px] font-semibold text-gray-200 tracking-wide">Snipping Tool</span>
            </div>
            <p className="text-[10px] text-gray-400 leading-snug">Artifact copied to clipboard. Automatically saved to recursive memory bank.</p>
            <button className="w-full mt-3 py-1.5 bg-neutral-800 text-white rounded text-[10px] hover:bg-neutral-700 transition-colors">Markup and share</button>
          </div>
        </div>
      </TiltCard>
    </ScrollReveal>
  </section>
);

const FeatureTwo = () => {
  return (
    <section className="py-32 container mx-auto px-6 md:px-12 grid lg:grid-cols-2 gap-16 items-center">
      <ScrollReveal delay={0} className="order-2 lg:order-1 h-[500px]">
        <TiltCard className="w-full h-full">
          <div className="bg-[#030303] border border-white/10 rounded-3xl w-full h-full flex items-center justify-center relative overflow-hidden p-8 shadow-2xl">
            {/* Remove previous gradient, let the Visualizer handle the ambient glow */}
            <div className="relative z-10 w-full h-full flex flex-col items-center justify-center">
              <CognitiveGraphVisualizer />
              <p className="mt-6 text-gray-500 text-[13px] leading-relaxed font-medium text-center px-4 max-w-[280px]">
                Active AGI topology: 7 core intelligence layers continuously operating in a cognitive loop.
              </p>
            </div>
          </div>
        </TiltCard>
      </ScrollReveal>

      <ScrollReveal delay={100} className="order-1 lg:order-2 max-w-md mx-auto lg:mx-0">
        <h2 className="text-4xl md:text-[52px] font-medium tracking-tight mb-8 text-white leading-[1.1]">
          <LetterReveal>Higher-level Abstractions</LetterReveal>
        </h2>
        <p className="text-gray-400 text-lg leading-relaxed">
          A more intuitive task-based approach to monitoring agent activity, presenting you with essential states and verification results to build trust.
        </p>
      </ScrollReveal>
    </section>
  );
};

const FeatureThree = () => {
  const icons = [Folder, Terminal, Command, Frame, Box, Code2, Code, Sparkles, Fingerprint, Wand2];
  return (
    <section className="py-40 container mx-auto px-6 text-center overflow-hidden">
      <div className="flex justify-center gap-4 md:gap-5 mb-24 py-4 flex-wrap max-w-4xl mx-auto">
        {icons.map((Icon, i) => (
          <ScrollReveal key={i} delay={i * 40}>
            <div
              className="w-[70px] h-[70px] rounded-full border border-white/10 bg-[#0a0a0b] flex items-center justify-center shrink-0 cursor-pointer shadow-xl transition-all duration-[280ms] ease-[cubic-bezier(0.22,1,0.36,1)] hover:scale-110 hover:-translate-y-2 hover:bg-neutral-900 hover:shadow-[0_10px_20px_rgba(255,255,255,0.05)]"
            >
              <Icon className="w-6 h-6 text-gray-400" strokeWidth={1.5} />
            </div>
          </ScrollReveal>
        ))}
      </div>
      <ScrollReveal delay={200} className="w-full relative px-4 flex justify-center">
        <h2 className="text-4xl md:text-[64px] font-semibold tracking-[-0.03em] leading-[1.05] text-white text-center max-w-[1000px]">
          <LetterReveal>
            UCGA is our graph-native<br />
            framework, evolving neural<br />
            networks into the AGI era.
          </LetterReveal>
          <span className="inline-block w-[3px] h-[45px] md:h-[60px] bg-gradient-to-b from-blue-400 to-red-400 ml-3 align-text-bottom mb-1 animate-pulse-slow" />
        </h2>
      </ScrollReveal>
    </section>
  );
};

const InteractiveSimulatorSection = () => (
  <section className="py-24 container mx-auto px-6 md:px-12 border-t border-white/5">
    <ScrollReveal className="max-w-3xl mx-auto text-center mb-16">
      <h2 className="text-4xl md:text-[52px] font-medium tracking-tight mb-6 text-white leading-[1.1]">
        <LetterReveal>Cognitive Test Simulator</LetterReveal>
      </h2>
      <p className="text-gray-400 text-lg leading-relaxed">
        Watch in real time how the UCGA Second Brain processes text, image, and video inputs, runs System 1 feature extraction, retrieves memories, reasons, and executes tools.
      </p>
    </ScrollReveal>
    <ScrollReveal delay={100}>
      <InteractiveSimulator />
    </ScrollReveal>
  </section>
);

const InfographicsSection = () => (
  <section className="py-24 container mx-auto px-6 md:px-12 border-t border-white/5">
    <ScrollReveal className="max-w-3xl mx-auto text-center mb-16">
      <h2 className="text-4xl md:text-[52px] font-medium tracking-tight mb-6 text-white leading-[1.1]">
        <LetterReveal>Architecture Infographics</LetterReveal>
      </h2>
      <p className="text-gray-400 text-lg leading-relaxed">
        Deep technical layout of the dual-process cognitive workflows and multimodal testing sequence.
      </p>
    </ScrollReveal>
    <div className="grid md:grid-cols-2 gap-12">
      <ScrollReveal delay={100}>
        <div className="bg-[#0b0b0d] border border-white/10 rounded-3xl p-6 shadow-2xl">
          <h3 className="text-lg font-semibold text-white mb-4">1. Dual-Process Cognitive Loop</h3>
          <div className="rounded-2xl overflow-hidden border border-white/5 bg-black/40 relative group">
            <img src="/second_brain_flow.png" alt="Second Brain Workflow" className="w-full h-auto object-cover transition-transform duration-500 group-hover:scale-105" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-6">
              <p className="text-xs text-gray-300">Illustrates episodic vector databases, semantic graph queries, thinking blocks, and tool executions.</p>
            </div>
          </div>
        </div>
      </ScrollReveal>
      <ScrollReveal delay={200}>
        <div className="bg-[#0b0b0d] border border-white/10 rounded-3xl p-6 shadow-2xl">
          <h3 className="text-lg font-semibold text-white mb-4">2. Multimodal Test Processing</h3>
          <div className="rounded-2xl overflow-hidden border border-white/5 bg-black/40 relative group">
            <img src="/multimodal_test.png" alt="Multimodal Testing" className="w-full h-auto object-cover transition-transform duration-500 group-hover:scale-105" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-6">
              <p className="text-xs text-gray-300">Shows Text, Image, and Video streams mapped through System 1 encoders into System 2 planning and critic loops.</p>
            </div>
          </div>
        </div>
      </ScrollReveal>
    </div>
  </section>
);

const App = () => {
  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30 font-sans relative">
      <StardustBackground />
      <div className="relative z-10">
        <Nav />
        <main>
          <Hero />
          <FeatureOne />
          <FeatureTwo />
          <FeatureThree />
          <InteractiveSimulatorSection />
          <InfographicsSection />
        </main>
      </div>
      <footer className="py-24 border-t border-white/5 mt-20 bg-black backdrop-blur-3xl relative z-10">
        <div className="container mx-auto px-6 grid md:grid-cols-4 gap-12 text-sm">
          <div className="md:col-span-1">
            <div className="flex items-center gap-3 mb-6">
              <img src="/ucga_logo.png" alt="UCGA Logo" className="h-8 md:h-10 w-auto object-contain" />
            </div>
          </div>
          <div>
            <h4 className="text-white font-medium mb-6">Product</h4>
            <ul className="space-y-4 text-gray-500">
              <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Integrations</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Changelog</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-6">Resources</h4>
            <ul className="space-y-4 text-gray-500">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Research Paper</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Community</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-6">Company</h4>
            <ul className="space-y-4 text-gray-500">
              <li><a href="#" className="hover:text-white transition-colors">About Us</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
            </ul>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
