import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import Papa from "papaparse";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  ScatterChart,
  Scatter
} from "recharts";
import { Upload, FileSpreadsheet, FileJson, Play, Pause, RotateCcw, Link as LinkIcon, Plus, Video } from "lucide-react";

/**
 * Scalable RL Portfolio — Data-Driven Viewer (+ Video)
 *
 * Drop in the three exported files and optional videos, or fetch from a repo/URL.
 *
 * Renders:
 * 1) learning_curves.csv → reward / success_rate / distil_loss / q_loss / critic_loss
 * 2) vector_field_bcflow.json → teacher BC-flow vector field (t slices)
 * 3) embedding_student_teacher.json → teacher vs student 2D embeddings
 * 4) videos (mp4/webm) → evaluation rollouts or training previews
 *
 * URL Loader supports:
 * - Direct file URLs (e.g., GitHub raw):
 *   https://raw.githubusercontent.com/<user>/<repo>/<branch>/exp/.../portfolio_logs/learning_curves.csv
 * - A "base URL" + quick-pick default filenames.
 */

// ------------------------------
// Helpers
// ------------------------------

async function readAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function classNames(...arr) {
  return arr.filter(Boolean).join(" ");
}

async function fetchText(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
  return await res.text();
}

// ------------------------------
// Learning Curves
// ------------------------------

function LearningCurves({ data }) {
  if (!data?.length) return null;
  return (
    <div className="w-full bg-gray-900 rounded-2xl shadow-xl p-6 text-gray-100">
      <h3 className="text-lg font-semibold mb-4">학습 곡선 — Reward / Success / Distillation / Q / Critic</h3>
      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="step" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <YAxis stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <Tooltip contentStyle={{ backgroundColor: "#0b0f19", border: "1px solid #1f2937", color: "#e5e7eb" }} />
            <Legend />
            <Line type="monotone" dataKey="reward" stroke="#60a5fa" dot={false} strokeWidth={2} name="Reward" />
            <Line type="monotone" dataKey="success_rate" stroke="#34d399" dot={false} strokeWidth={2} name="Success" />
            <Line type="monotone" dataKey="distil_loss" stroke="#f472b6" dot={false} strokeWidth={2} name="Distil" />
            <Line type="monotone" dataKey="bc_flow_loss" stroke="#22d3ee" dot={false} strokeWidth={1.8} name="BC-Flow" />
            <Line type="monotone" dataKey="q_loss" stroke="#fbbf24" dot={false} strokeWidth={1.8} name="Q-Loss" />
            <Line type="monotone" dataKey="critic_loss" stroke="#a3e635" dot={false} strokeWidth={1.5} name="Critic" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ------------------------------
// Vector Field Canvas (Teacher BC-flow)
// ------------------------------

function VectorFieldCanvas({ vf }) {
  const canvasRef = useRef(null);
  const [tIndex, setTIndex] = useState(0);
  const [running, setRunning] = useState(false);

  const slices = vf?.vector_field ?? [];
  const current = slices[tIndex];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !current) return;
    const ctx = canvas.getContext("2d");

    const onResize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      draw();
    };

    const xs = current.points.map(p => p[0]);
    const ys = current.points.map(p => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);

    const toCanvas = (x, y) => {
      const w = canvas.width, h = canvas.height;
      const nx = (x - xmin) / Math.max(1e-6, (xmax - xmin)) * (w * 0.9) + w * 0.05;
      const ny = h - ((y - ymin) / Math.max(1e-6, (ymax - ymin)) * (h * 0.9) + h * 0.05);
      return [nx, ny];
    };

    function draw() {
      ctx.fillStyle = "#0b0f19";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!current) return;
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 1;

      for (let i = 0; i < current.points.length; i++) {
        const [x, y] = current.points[i];
        const [vx, vy] = current.vectors[i];
        const [sx, sy] = toCanvas(x, y);
        const [ex, ey] = toCanvas(x + vx, y + vy);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
      }

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px ui-sans-serif";
      ctx.fillText(`t = ${current.t}`, 12, 18);
    }

    onResize();
    window.addEventListener("resize", onResize);
    draw();
    return () => window.removeEventListener("resize", onResize);
  }, [current]);

  useEffect(() => {
    if (!running || slices.length <= 1) return;
    const id = setInterval(() => setTIndex(i => (i + 1) % slices.length), 800);
    return () => clearInterval(id);
  }, [running, slices.length]);

  if (!slices.length) return null;

  return (
    <div className="w-full bg-gray-900 rounded-2xl shadow-xl p-4 text-gray-100">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-gray-100 font-semibold">BC-flow 벡터필드 (Teacher)</h3>
        <div className="flex gap-2">
          <button onClick={() => setRunning(r => !r)} className="px-3 py-1.5 rounded-xl bg-gray-800 hover:bg-gray-700 flex items-center gap-1">
            {running ? <Pause size={16}/> : <Play size={16}/>} {running ? "Pause" : "Play"}
          </button>
          <button onClick={() => setTIndex(0)} className="px-3 py-1.5 rounded-xl bg-gray-800 hover:bg-gray-700 flex items-center gap-1">
            <RotateCcw size={16}/> Reset
          </button>
        </div>
      </div>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-sm text-gray-300">t slice</span>
        <input type="range" min={0} max={slices.length - 1} value={tIndex} onChange={e => setTIndex(parseInt(e.target.value))} className="w-56" />
        <span className="text-sm text-gray-400">{slices[tIndex]?.t}</span>
      </div>
      <div className="w-full h-[360px] rounded-xl overflow-hidden border border-gray-800">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
      <p className="text-xs text-gray-400 mt-2">각 점의 화살표는 (x,y)에서의 예측 속도(∂x/∂t)를 나타냅니다.</p>
    </div>
  );
}

// ------------------------------
// Embedding Scatter (Teacher vs Student)
// ------------------------------

function EmbeddingScatter({ emb }) {
  if (!emb?.teacher || !emb?.student) return null;
  const teacher = emb.teacher.map((p, i) => ({ x: p[0], y: p[1], id: `T${i}` }));
  const student = emb.student.map((p, i) => ({ x: p[0], y: p[1], id: `S${i}` }));

  return (
    <div className="w-full bg-gray-900 rounded-2xl shadow-xl p-6 text-gray-100">
      <h3 className="text-lg font-semibold mb-4">Action Embedding — Teacher vs Student</h3>
      <div className="w-full h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 24, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#1f2937" />
            <XAxis type="number" dataKey="x" name="x" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <YAxis type="number" dataKey="y" name="y" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ backgroundColor: "#0b0f19", border: "1px solid #1f2937", color: "#e5e7eb" }} />
            <Legend />
            <Scatter name="Teacher" data={teacher} fill="#22d3ee" />
            <Scatter name="Student" data={student} fill="#a78bfa" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-gray-400">동일 관측에서의 행동 분포 비교. student가 teacher 분포로 수렴하는지 확인하세요.</p>
    </div>
  );
}

// ------------------------------
// Video Gallery
// ------------------------------

function VideoGallery({ videos }) {
  if (!videos?.length) return null;
  return (
    <div className="w-full bg-gray-900 rounded-2xl shadow-xl p-6 text-gray-100">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Video size={18}/>Evaluation / Demo Videos</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {videos.map((v, idx) => (
          <div key={idx} className="bg-black/40 border border-gray-800 rounded-xl overflow-hidden">
            <video src={v.src} controls className="w-full h-64 object-contain bg-black" />
            <div className="px-3 py-2 text-xs text-gray-300 border-t border-gray-800 flex items-center justify-between">
              <span className="truncate" title={v.name}>{v.name}</span>
              {v.origin && <span className="text-[10px] text-gray-500">{v.origin}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ------------------------------
// File/URL Loaders
// ------------------------------

function DropLoader({ onFiles }) {
  const [drag, setDrag] = useState(false);

  const onDrop = (ev) => {
    ev.preventDefault();
    setDrag(false);
    const files = Array.from(ev.dataTransfer.files || []);
    onFiles(files);
  };
  const onDragOver = (ev) => { ev.preventDefault(); setDrag(true); };
  const onDragLeave = () => setDrag(false);

  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={classNames(
        "w-full border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer",
        drag ? "border-cyan-400 bg-cyan-900/10" : "border-gray-700 bg-gray-900"
      )}
    >
      <div className="flex flex-col items-center gap-2 text-gray-200">
        <Upload />
        <div className="text-sm">여기에 파일을 드롭하거나, 아래에서 개별 선택/URL 불러오기를 사용하세요.</div>
        <div className="text-xs text-gray-400">CSV/JSON/MP4(WebM) 지원</div>
      </div>
    </div>
  );
}

function UrlLoader({ onLoadQuick, onLoadDirect }) {
  const [baseUrl, setBaseUrl] = useState("");
  const [directUrl, setDirectUrl] = useState("");

  const defaults = [
    "portfolio_logs/learning_curves.csv",
    "portfolio_logs/vector_field_bcflow.json",
    "portfolio_logs/embedding_student_teacher.json",
  ];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 text-gray-200 space-y-3">
      <div className="font-semibold text-sm mb-1 flex items-center gap-2"><LinkIcon size={16}/>URL 불러오기</div>
      <div className="text-xs text-gray-400">GitHub raw 또는 임의의 정적 파일 서버에서 직접 불러올 수 있습니다.</div>

      <div className="flex flex-col gap-2">
        <label className="text-xs text-gray-400">Base URL (옵션) — 예: https://raw.githubusercontent.com/&lt;user&gt;/&lt;repo&gt;/&lt;branch&gt;/exp/.../</label>
        <input className="bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-sm" placeholder="https://raw.githubusercontent.com/user/repo/branch/path/to/run/" value={baseUrl} onChange={(e)=>setBaseUrl(e.target.value)} />
        <div className="flex flex-wrap gap-2">
          {defaults.map((p)=> (
            <button key={p} onClick={()=> onLoadQuick(baseUrl, p)} className="px-3 py-1.5 text-xs rounded-xl bg-gray-800 hover:bg-gray-700 border border-gray-700">{p}</button>
          ))}
        </div>
      </div>

      <div className="h-px bg-gray-800" />

      <div className="flex items-center gap-2">
        <input className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-sm" placeholder="직접 URL (CSV/JSON/MP4/WebM)" value={directUrl} onChange={(e)=>setDirectUrl(e.target.value)} />
        <button onClick={()=> onLoadDirect(directUrl)} className="px-3 py-2 text-sm rounded-xl bg-cyan-700 hover:bg-cyan-600">불러오기</button>
      </div>
    </div>
  );
}

// ------------------------------
// Root Component
// ------------------------------

export default function DataDrivenPortfolio() {
  const [curves, setCurves] = useState(null);
  const [vf, setVf] = useState(null);
  const [emb, setEmb] = useState(null);
  const [videos, setVideos] = useState([]); // {name, src, origin?}

  const handleFiles = async (files) => {
    for (const f of files) {
      if (!f) continue;
      const name = f.name.toLowerCase();
      if (f.type.startsWith("video/")) {
        const url = URL.createObjectURL(f);
        setVideos(v => [...v, { name: f.name, src: url, origin: "local" }]);
      } else if (name.endsWith(".csv") && name.includes("learning_curves")) {
        const text = await readAsText(f);
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true });
        const rows = (parsed.data || []).filter(r => r && r.step !== undefined);
        setCurves(rows);
      } else if (name.endsWith(".json") && name.includes("vector_field")) {
        const text = await readAsText(f);
        setVf(JSON.parse(text));
      } else if (name.endsWith(".json") && name.includes("embedding")) {
        const text = await readAsText(f);
        setEmb(JSON.parse(text));
      }
    }
  };

  const loadQuick = async (baseUrl, relPath) => {
    if (!baseUrl) return;
    const url = baseUrl.replace(/\/+$/, "") + "/" + relPath;
    await loadDirect(url);
  };

  const loadDirect = async (url) => {
    if (!url) return;
    const lower = url.toLowerCase();
    try {
      if (lower.endsWith(".csv")) {
        const text = await fetchText(url);
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true });
        const rows = (parsed.data || []).filter(r => r && r.step !== undefined);
        setCurves(rows);
      } else if (lower.endsWith(".json") && lower.includes("vector_field")) {
        const text = await fetchText(url);
        setVf(JSON.parse(text));
      } else if (lower.endsWith(".json") && lower.includes("embedding")) {
        const text = await fetchText(url);
        setEmb(JSON.parse(text));
      } else if (lower.endsWith(".mp4") || lower.endsWith(".webm")) {
        setVideos(v => [...v, { name: url.split("/").pop() || "video", src: url, origin: "url" }]);
      } else {
        alert("지원하지 않는 파일 형식입니다. CSV/JSON/MP4/WebM 만 가능합니다.");
      }
    } catch (e) {
      alert(`불러오기 실패: ${e.message}`);
    }
  };

  const onPickCsv = (e) => handleFiles([e.target.files?.[0]]);
  const onPickVf = (e) => handleFiles([e.target.files?.[0]]);
  const onPickEmb = (e) => handleFiles([e.target.files?.[0]]);
  const onPickVideo = (e) => handleFiles(Array.from(e.target.files || []));

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-gray-950 to-gray-900 text-gray-100">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <motion.h1 initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="text-2xl md:text-4xl font-bold tracking-tight">
          Scalable RL Portfolio — <span className="text-cyan-300">Data-Driven + Video</span>
        </motion.h1>
        <p className="text-gray-300 mt-2">내보낸 파일 3개와 선택 영상들을 불러와 즉시 시각화합니다. GitHub raw URL도 지원합니다.</p>

        <div className="mt-6">
          <DropLoader onFiles={handleFiles} />
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
          <label className="bg-gray-900 border border-gray-700 rounded-xl p-4 flex items-center gap-3 cursor-pointer hover:border-cyan-500">
            <FileSpreadsheet />
            <div>
              <div className="text-sm">learning_curves.csv</div>
              <div className="text-xs text-gray-400">학습 곡선</div>
            </div>
            <input type="file" accept=".csv" className="hidden" onChange={onPickCsv} />
          </label>
          <label className="bg-gray-900 border border-gray-700 rounded-xl p-4 flex items-center gap-3 cursor-pointer hover:border-cyan-500">
            <FileJson />
            <div>
              <div className="text-sm">vector_field_bcflow.json</div>
              <div className="text-xs text-gray-400">BC-flow 벡터필드</div>
            </div>
            <input type="file" accept="application/json" className="hidden" onChange={onPickVf} />
          </label>
          <label className="bg-gray-900 border border-gray-700 rounded-xl p-4 flex items-center gap-3 cursor-pointer hover:border-cyan-500">
            <FileJson />
            <div>
              <div className="text-sm">embedding_student_teacher.json</div>
              <div className="text-xs text-gray-400">Teacher vs Student</div>
            </div>
            <input type="file" accept="application/json" className="hidden" onChange={onPickEmb} />
          </label>
          <label className="bg-gray-900 border border-gray-700 rounded-xl p-4 flex items-center gap-3 cursor-pointer hover:border-cyan-500">
            <Video />
            <div>
              <div className="text-sm">Videos (MP4/WebM)</div>
              <div className="text-xs text-gray-400">평가/데모 영상 여러 개 선택 가능</div>
            </div>
            <input type="file" accept="video/*" className="hidden" multiple onChange={onPickVideo} />
          </label>
        </div>

        <div className="mt-4">
          <UrlLoader onLoadQuick={loadQuick} onLoadDirect={loadDirect} />
        </div>

        {/* Render sections when data available */}
        <div className="mt-6 space-y-6">
          <LearningCurves data={curves} />
          <VectorFieldCanvas vf={vf} />
          <EmbeddingScatter emb={emb} />
          <VideoGallery videos={videos} />
        </div>

        {/* Tips */}
        <div className="mt-10 bg-gray-900 rounded-2xl shadow-xl p-6 text-sm text-gray-300">
          <div className="font-semibold mb-2">데이터/동영상 포맷 체크리스트</div>
          <ul className="list-disc pl-6 space-y-1">
            <li><code className="bg-gray-800 px-1 rounded">learning_curves.csv</code>: step은 정수, 나머지는 숫자.</li>
            <li><code className="bg-gray-800 px-1 rounded">vector_field_bcflow.json</code>: 동일 길이의 <em>points</em>/<em>vectors</em>.</li>
            <li><code className="bg-gray-800 px-1 rounded">embedding_student_teacher.json</code>: teacher/student 각각 [[x,y],...].</li>
            <li>비디오는 <code className="bg-gray-800 px-1 rounded">.mp4</code> 또는 <code className="bg-gray-800 px-1 rounded">.webm</code> 권장.</li>
            <li>GitHub는 <span className="bg-gray-800 px-1 rounded">raw</span> URL을 사용하세요 (권한 필요 없음).</li>
          </ul>
        </div>

        <footer className="text-xs text-gray-500 mt-8">© {new Date().getFullYear()} Scalable RL Portfolio</footer>
      </div>
    </div>
  );
}
