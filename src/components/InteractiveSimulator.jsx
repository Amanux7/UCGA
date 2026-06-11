import React, { useState, useEffect } from 'react';
import { 
  Play, RotateCcw, AlertCircle, CheckCircle, Database, Network, Cpu, FileCode2,
  Image as ImageIcon, Video as VideoIcon, FileText, Check, ArrowRight, Pause
} from 'lucide-react';

// Sample test cases for the simulator
const TEST_CASES = {
  text: {
    icon: FileText,
    label: "Text Reasoning Task",
    query: "Compare UCGA performance data with standard MLPs and compute the ratio of parameters.",
    system1: "Encoding text syntax. Sentence embedding dimension: [1, 128].",
    retrieved: {
      episodic: [
        "Memory 1: UCGA parameters = 120K. MLP parameters = 450K.",
        "Memory 2: Accuracy comparison: UCGA achieves 92%, MLP achieves 85%."
      ],
      semantic: [
        "UCGA -[compares_to]-> MLP",
        "UCGA -[has_parameters]-> 120K",
        "MLP -[has_parameters]-> 450K"
      ]
    },
    thinking: "I need to calculate the parameter ratio: UCGA (120K) to MLP (450K). I will use the python_interpreter tool to execute 120 / 450.",
    tool: "python_interpreter",
    toolArg: '{"code": "print(round(120 / 450, 4))"}',
    toolResult: "0.2667",
    criticStatus: "APPROVED",
    criticIssues: "None",
    criticSuggestions: "Output the final ratio directly as a percentage in final_output.",
    finalOutput: "Based on episodic memory, UCGA uses 120K parameters while standard MLPs use 450K. The parameters ratio is 0.2667, meaning UCGA requires only 26.67% of the parameters of a standard MLP while achieving higher accuracy."
  },
  image: {
    icon: ImageIcon,
    label: "Image Reasoning Task",
    query: "Detect anomalies in the input circuit diagram, retrieve corrective actions, and log to circuits.log.",
    system1: "Processing image tensor [3, 256, 256]. Extracting CNN spatial features. Detecting hot spot at pixel coords (142, 88).",
    retrieved: {
      episodic: [
        "Memory 1: Overheating at coords (142, 88) usually indicates a blown resistor R5.",
        "Memory 2: Standard corrective action: write R5 fuse shutdown script to log."
      ],
      semantic: [
        "CircuitBoard -[contains]-> Resistor_R5",
        "Resistor_R5 -[status]-> Blown",
        "Blown -[requires_action]-> ShutdownLog"
      ]
    },
    thinking: "Spatial features detect hot spot at R5. Epistemic memory recommends logging a shutdown sequence for R5 to circuits.log. I will call the file_writer tool.",
    tool: "file_writer",
    toolArg: '{"filename": "circuits.log", "content": "SHUTDOWN SEQUENCE INITIATED: Blown resistor R5 detected at (142, 88)"}',
    toolResult: "File written successfully: data/second_brain/files/circuits.log",
    criticStatus: "APPROVED",
    criticSuggestions: "Verify log path confirmation.",
    finalOutput: "Anomaly detected: blown Resistor R5 at pixel coordinates (142, 88). Corrective action has been written to data/second_brain/files/circuits.log successfully."
  },
  video: {
    icon: VideoIcon,
    label: "Video Frame Sequence",
    query: "Track object trajectory across 24 video frames and compute the acceleration.",
    system1: "Processing video sequence: 24 frames of size 640x480. Extracting temporal optical flow. Tracking bounding box [x:12, y:45, w:30, h:30] at 30 fps.",
    retrieved: {
      episodic: [
        "Memory 1: Moving object is classified as target drone model T-100.",
        "Memory 2: Position trajectory logged: [t=0s, x=0m], [t=0.4s, x=2m], [t=0.8s, x=8m]."
      ],
      semantic: [
        "Drone_T100 -[has_optical_signature]-> DynamicFlow",
        "Trajectory -[measures]-> Acceleration"
      ]
    },
    thinking: "Object coordinates: t=0: 0m, t=0.4: 2m, t=0.8: 8m. Position follows x = 12.5 * t^2. Double derivative gives acceleration a = 25 m/s^2. I will run the python interpreter to double check.",
    tool: "python_interpreter",
    toolArg: '{"code": "t1, x1 = 0.4, 2.0\\nt2, x2 = 0.8, 8.0\\n# x = 0.5 * a * t^2\\na1 = 2 * x1 / (t1**2)\\na2 = 2 * x2 / (t2**2)\\nprint(f\'a1={a1}, a2={a2}\')"}',
    toolResult: "a1=25.0, a2=25.0",
    criticStatus: "APPROVED",
    criticSuggestions: "Confirm classification model name in final output.",
    finalOutput: "Video sequence tracks drone model T-100. From optical flow coordinates, position is modeled as x(t) = 12.5*t^2. Computing second derivative yields a constant acceleration of 25.0 m/s^2."
  }
};

export default function InteractiveSimulator() {
  const [activeTab, setActiveTab] = useState("text");
  const [isPlaying, setIsPlaying] = useState(false);
  const [simStep, setSimStep] = useState(0); // 0: Idle, 1: System 1, 2: Context retrieval, 3: Reasoning, 4: Critic, 5: Tool call, 6: Output
  const [simProgress, setSimProgress] = useState(0);
  const [typewriterText, setTypewriterText] = useState("");

  const testCase = TEST_CASES[activeTab];

  // Steps definition
  const steps = [
    { label: "Idle", desc: "Ready to run test query" },
    { label: "System 1 Encoders", desc: "Sensory Feature Extraction" },
    { label: "Context Ingestion", desc: "Episodic & Semantic Lookup" },
    { label: "Chain-of-Thought", desc: "System 2 Deliberate Reasoning" },
    { label: "Metacognitive Critique", desc: "Metacognitive Self-Correction" },
    { label: "Tool Execution", desc: "Procedural Tool Call" },
    { label: "Final Output", desc: "Response Generation" }
  ];

  // Reset simulator
  const handleReset = () => {
    setIsPlaying(false);
    setSimStep(0);
    setSimProgress(0);
    setTypewriterText("");
  };

  // Run or Pause simulation
  const handleTogglePlay = () => {
    if (simStep === 6) {
      // Loop back if finished
      setSimStep(0);
      setSimProgress(0);
      setTypewriterText("");
      setIsPlaying(true);
    } else {
      setIsPlaying(!isPlaying);
    }
  };

  // Handle tab change
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    handleReset();
  };

  // Auto-advance simulation when playing
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSimStep((prevStep) => {
          const nextStep = prevStep + 1;
          if (nextStep > 6) {
            setIsPlaying(false);
            return 6;
          }
          setSimProgress((nextStep / 6) * 100);
          return nextStep;
        });
      }, 3000); // 3 seconds per step
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Handle typewriter effect for thinking block
  useEffect(() => {
    if (simStep === 3) {
      let currentLength = 0;
      const fullText = testCase.thinking;
      const typewriterInterval = setInterval(() => {
        if (currentLength < fullText.length) {
          setTypewriterText(fullText.substring(0, currentLength + 2));
          currentLength += 2;
        } else {
          clearInterval(typewriterInterval);
        }
      }, 15);
      return () => clearInterval(typewriterInterval);
    }
  }, [simStep, testCase.thinking]);

  return (
    <div className="w-full bg-[#050505] border border-white/10 rounded-3xl p-6 md:p-8 shadow-2xl relative overflow-hidden backdrop-blur-md">
      {/* Glow Effects */}
      <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-blue-500/10 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full bg-purple-500/10 blur-3xl pointer-events-none" />

      {/* Header & Tabs */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8 border-b border-white/10 pb-6 relative z-10">
        <div>
          <h3 className="text-xl md:text-2xl font-semibold text-white tracking-tight flex items-center gap-2">
            <Cpu className="text-blue-400 w-5 h-5 animate-pulse" />
            Dual-Process Cognitive Loop Simulator
          </h3>
          <p className="text-gray-400 text-sm mt-1">Select a sensory input modality to simulate the System 1/2 cognitive pass.</p>
        </div>

        <div className="flex bg-[#0f0f11] border border-white/5 rounded-full p-1 self-stretch md:self-auto justify-around">
          {Object.keys(TEST_CASES).map((tab) => {
            const TabIcon = TEST_CASES[tab].icon;
            const isActive = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => handleTabChange(tab)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold tracking-wider transition-all duration-300 ${
                  isActive 
                    ? "bg-white text-black shadow-lg" 
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <TabIcon className="w-3.5 h-3.5" />
                {TEST_CASES[tab].label.split(" ")[0]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Visual Simulation Display Area */}
      <div className="grid lg:grid-cols-3 gap-8 relative z-10">
        
        {/* Left Side: Test Input Modality Preview & Flow Controls */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          
          {/* Input Panel Card */}
          <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 flex-1 flex flex-col">
            <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 block">Test Input</span>
            <div className="bg-[#121215] border border-white/5 rounded-xl p-4 flex-1 flex flex-col justify-center items-center relative overflow-hidden group">
              
              {/* Modality Specific Mock Graphics */}
              {activeTab === "text" && (
                <div className="w-full text-center">
                  <FileText className="w-16 h-16 text-blue-500/80 mx-auto mb-3 animate-float" />
                  <p className="text-xs text-gray-300 font-semibold px-2">"{testCase.query}"</p>
                </div>
              )}

              {activeTab === "image" && (
                <div className="w-full flex flex-col items-center">
                  <div className="w-40 h-28 border border-dashed border-white/20 rounded-lg flex items-center justify-center relative overflow-hidden bg-black/40">
                    <ImageIcon className="w-8 h-8 text-cyan-500/60" />
                    {simStep >= 1 && (
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-red-500/30 rounded-full border border-red-500 animate-ping" />
                    )}
                    {simStep >= 1 && (
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-red-500 rounded-full" />
                    )}
                  </div>
                  <span className="text-[10px] text-gray-500 mt-2">Circuits schematic R5 grid anomaly test</span>
                </div>
              )}

              {activeTab === "video" && (
                <div className="w-full flex flex-col items-center">
                  <div className="w-40 h-28 border border-white/10 rounded-lg flex items-center justify-center relative overflow-hidden bg-black/40">
                    <VideoIcon className="w-8 h-8 text-purple-500/60" />
                    {/* Bounding box animation */}
                    {simStep >= 1 && (
                      <div className="absolute border border-green-400 bg-green-400/10 w-8 h-8 rounded-sm animate-pulse" 
                        style={{
                          left: isPlaying ? `${15 + simStep * 6}%` : '40%',
                          top: '35%'
                        }}
                      />
                    )}
                    <div className="absolute bottom-2 left-2 right-2 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-purple-500 animate-pulse" style={{ width: isPlaying ? `${(simStep / 6) * 100}%` : '50%' }} />
                    </div>
                  </div>
                  <span className="text-[10px] text-gray-500 mt-2">Drone Tracking Frame sequence (t=0..0.8s)</span>
                </div>
              )}

            </div>

            {/* Simulation controls */}
            <div className="mt-5 flex gap-3">
              <button
                onClick={handleTogglePlay}
                className={`flex-1 py-3 px-4 rounded-xl flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider transition-all duration-300 ${
                  isPlaying 
                    ? "bg-amber-500 text-black hover:bg-amber-400" 
                    : "bg-white text-black hover:bg-neutral-200"
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-black" /> : <Play className="w-3.5 h-3.5 fill-black" />}
                {isPlaying ? "Pause" : simStep === 6 ? "Restart" : "Simulate Test"}
              </button>
              
              <button
                onClick={handleReset}
                className="p-3 bg-neutral-900 border border-white/10 text-white rounded-xl hover:bg-neutral-800 transition-colors"
                title="Reset simulation"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Timeline Process Steps */}
          <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 flex flex-col gap-3">
            <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-1">Execution Pipeline</span>
            <div className="flex flex-col gap-3 relative before:absolute before:left-3.5 before:top-2 before:bottom-2 before:w-[1px] before:bg-white/15">
              {steps.map((step, idx) => {
                if (idx === 0) return null; // Hide idle step from list
                const isCurrent = simStep === idx;
                const isCompleted = simStep > idx;
                return (
                  <div key={idx} className="flex gap-4 items-start relative">
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold z-10 border transition-all duration-300 ${
                      isCurrent 
                        ? "bg-blue-500 border-blue-400 text-white shadow-[0_0_12px_rgba(59,130,246,0.5)]" 
                        : isCompleted
                          ? "bg-green-600/20 border-green-500 text-green-400"
                          : "bg-[#121215] border-white/15 text-gray-500"
                    }`}>
                      {isCompleted ? <Check className="w-3.5 h-3.5" /> : idx}
                    </div>
                    <div>
                      <h4 className={`text-xs font-semibold transition-colors duration-300 ${isCurrent ? "text-white" : isCompleted ? "text-gray-300" : "text-gray-500"}`}>
                        {step.label}
                      </h4>
                      <p className={`text-[10px] transition-colors duration-300 ${isCurrent ? "text-blue-400" : "text-gray-500"}`}>
                        {step.desc}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

        {/* Right Side (Center & Right Column combined): Realtime State & Active Cognitive Workspace */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* Top Panel: System 1 Encoders & Retrieval */}
          <div className="grid md:grid-cols-2 gap-6">
            
            {/* System 1 Card */}
            <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 min-h-[160px] flex flex-col">
              <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" />
                System 1 Sensory Encoders
              </span>
              <div className="flex-1 bg-[#121215] border border-white/5 rounded-xl p-4 font-mono text-[11px] leading-relaxed text-gray-400 overflow-y-auto">
                {simStep >= 1 ? (
                  <span className="text-gray-300 animate-pulse">{testCase.system1}</span>
                ) : (
                  <span className="text-gray-600 italic">// Waiting to receive sensor stream...</span>
                )}
              </div>
            </div>

            {/* Memory Layer Retrieval Card */}
            <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 min-h-[160px] flex flex-col">
              <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5" />
                Context Retrieval (Hybrid LTM)
              </span>
              <div className="flex-1 bg-[#121215] border border-white/5 rounded-xl p-4 font-mono text-[10px] leading-relaxed text-gray-400 overflow-y-auto flex flex-col gap-2">
                {simStep >= 2 ? (
                  <>
                    <div className="flex flex-col gap-1">
                      <span className="text-blue-400 text-[9px] font-bold uppercase">📥 Episodic Memory (Vector DB):</span>
                      {testCase.retrieved.episodic.map((m, i) => (
                        <span key={i} className="text-gray-300 pl-2 border-l border-blue-500/30">{m}</span>
                      ))}
                    </div>
                    <div className="flex flex-col gap-1 mt-1">
                      <span className="text-yellow-500 text-[9px] font-bold uppercase">🕸️ Semantic Memory (Knowledge Graph):</span>
                      {testCase.retrieved.semantic.map((m, i) => (
                        <span key={i} className="text-gray-300 pl-2 border-l border-yellow-500/30">{m}</span>
                      ))}
                    </div>
                  </>
                ) : (
                  <span className="text-gray-600 italic">// Inactive. Trigger to query vector/graph index...</span>
                )}
              </div>
            </div>

          </div>

          {/* Core Reasoning Console */}
          <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 flex-1 flex flex-col min-h-[300px]">
            <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 flex items-center gap-1.5">
              <FileCode2 className="w-3.5 h-3.5" />
              Active System 2 Reasoning & Metacognitive Critic
            </span>
            <div className="flex-1 flex flex-col md:grid md:grid-cols-2 gap-4">
              
              {/* Agent Thinking Box */}
              <div className="bg-[#121215] border border-white/5 rounded-xl p-4 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest">🧠 Cognitive Chain-of-Thought</span>
                    {simStep >= 3 && <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-ping" />}
                  </div>
                  <div className="font-mono text-[11px] leading-relaxed text-gray-300 min-h-[140px] whitespace-pre-line">
                    {simStep >= 3 ? (
                      <>
                        <span className="text-blue-400">&lt;thinking&gt;</span>
                        <br />
                        {typewriterText}
                        {simStep === 3 && <span className="inline-block w-1.5 h-3.5 bg-white ml-0.5 animate-pulse" />}
                        {simStep > 3 && (
                          <>
                            <br />
                            <span className="text-blue-400">&lt;/thinking&gt;</span>
                          </>
                        )}
                      </>
                    ) : (
                      <span className="text-gray-600 italic">// Pending System 2 deliberative thought sequence...</span>
                    )}
                  </div>
                </div>
                {simStep >= 3 && (
                  <div className="mt-3 border-t border-white/5 pt-3 flex items-center justify-between text-[10px]">
                    <span className="text-gray-500">Routing to evaluation node...</span>
                    <span className="text-blue-400 font-semibold">T=1 passes</span>
                  </div>
                )}
              </div>

              {/* Critic Evaluation Box */}
              <div className="bg-[#121215] border border-white/5 rounded-xl p-4 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-[9px] font-bold text-gray-400 uppercase tracking-widest">🔍 Metacognitive Critic</span>
                    {simStep === 4 && <span className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-ping" />}
                  </div>
                  <div className="font-mono text-[10px] leading-relaxed text-gray-300 min-h-[140px] flex flex-col gap-2">
                    {simStep >= 4 ? (
                      <>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-400">Status:</span>
                          <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                            testCase.criticStatus === "APPROVED" 
                              ? "bg-green-500/10 text-green-400 border border-green-500/20" 
                              : "bg-red-500/10 text-red-400 border border-red-500/20"
                          }`}>
                            {testCase.criticStatus}
                          </span>
                        </div>
                        {testCase.criticIssues !== "None" && (
                          <div>
                            <span className="text-gray-400 block">Identified Issues:</span>
                            <span className="text-red-400 pl-2">• {testCase.criticIssues}</span>
                          </div>
                        )}
                        <div>
                          <span className="text-gray-400 block">Suggestions:</span>
                          <span className="text-purple-400 pl-2 animate-pulse">• {testCase.criticSuggestions}</span>
                        </div>
                      </>
                    ) : (
                      <span className="text-gray-600 italic">// Inactive. Waiting for draft solution feedback...</span>
                    )}
                  </div>
                </div>
                {simStep >= 4 && (
                  <div className="mt-3 border-t border-white/5 pt-3 flex items-center gap-2 text-[10px]">
                    {testCase.criticStatus === "APPROVED" ? (
                      <>
                        <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                        <span className="text-green-400 font-semibold">Self-correction approved.</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-3.5 h-3.5 text-red-500" />
                        <span className="text-red-400 font-semibold">Self-correction triggered.</span>
                      </>
                    )}
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* Bottom Panel: Tool Call & Output Result */}
          <div className="grid md:grid-cols-3 gap-6">
            
            {/* Tool Dispatch Panel */}
            <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 md:col-span-1 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 block">Procedural Tool Call</span>
                <div className="bg-[#121215] border border-white/5 rounded-xl p-3 font-mono text-[10px] leading-relaxed text-gray-400 min-h-[70px]">
                  {simStep >= 5 ? (
                    <>
                      <span className="text-blue-400">tool:</span> {testCase.tool}
                      <br />
                      <span className="text-blue-400">args:</span> {testCase.toolArg}
                    </>
                  ) : (
                    <span className="text-gray-600 italic">// No active tool call dispatch...</span>
                  )}
                </div>
              </div>
              {simStep >= 5 && (
                <div className="mt-3 text-[10px] font-mono text-gray-500">
                  <span className="text-green-500">Result:</span> {testCase.toolResult}
                </div>
              )}
            </div>

            {/* Final Answer Panel */}
            <div className="bg-[#0b0b0d] border border-white/5 rounded-2xl p-5 md:col-span-2 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-bold text-blue-400 tracking-widest uppercase mb-3 block">Final Cognitive Output</span>
                <div className="bg-[#121215] border border-white/5 rounded-xl p-3 font-mono text-[11px] leading-relaxed text-gray-300 min-h-[70px]">
                  {simStep >= 6 ? (
                    <>
                      <span className="text-green-400">&lt;final_output&gt;</span>
                      <br />
                      {testCase.finalOutput}
                      <br />
                      <span className="text-green-400">&lt;/final_output&gt;</span>
                    </>
                  ) : (
                    <span className="text-gray-600 italic">// Waiting to generate final output...</span>
                  )}
                </div>
              </div>
              {simStep >= 6 && (
                <div className="mt-3 flex items-center justify-between text-[10px] text-gray-500">
                  <span>Loop Execution Time: ~1.2s</span>
                  <span className="text-green-400 font-bold flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5 fill-green-500/10 text-green-400" />
                    Success
                  </span>
                </div>
              )}
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
