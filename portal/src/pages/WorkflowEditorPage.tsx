import { useState, useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  Handle,
  Position,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { HudTopBar } from "@/components/hud";
import { plans } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Custom Node                                                         */
/* ------------------------------------------------------------------ */

interface ActionNodeData {
  label: string;
  skillName: string;
  description: string;
  riskLevel: string;
  [key: string]: unknown;
}

function ActionNode({ data, selected }: { data: ActionNodeData; selected?: boolean }) {
  const riskColors: Record<string, string> = {
    safe: "border-hud-success/40",
    low: "border-hud-info/40",
    medium: "border-hud-warning/40",
    high: "border-hud-error/40",
    critical: "border-red-500/60",
  };

  return (
    <div
      className={`rounded-lg border-2 bg-hud-bg-medium px-4 py-3 min-w-[180px] shadow-lg transition-all ${
        selected ? "border-hud-accent shadow-glow" : riskColors[data.riskLevel] || "border-white/10"
      }`}
    >
      <Handle type="target" position={Position.Top} className="!bg-hud-accent3 !w-3 !h-3 !border-2 !border-hud-bg-dark" />
      <div className="text-sm font-medium text-white mb-1">{data.label}</div>
      {data.skillName && (
        <div className="text-[10px] font-mono text-hud-accent/70">{data.skillName}</div>
      )}
      {data.description && (
        <div className="text-[10px] text-white/30 mt-1 max-w-[160px] truncate">{data.description}</div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-hud-accent !w-3 !h-3 !border-2 !border-hud-bg-dark" />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  action: ActionNode,
};

/* ------------------------------------------------------------------ */
/* Node Palette (drag source)                                          */
/* ------------------------------------------------------------------ */

const PALETTE_ITEMS = [
  { label: "LLM Generate", skillName: "llm_generate", riskLevel: "low", icon: "🤖" },
  { label: "Web Search", skillName: "web_search", riskLevel: "low", icon: "🔍" },
  { label: "Code Generate", skillName: "code_generate", riskLevel: "medium", icon: "💻" },
  { label: "Docker Build", skillName: "docker_build", riskLevel: "medium", icon: "🐳" },
  { label: "Deploy", skillName: "deploy", riskLevel: "high", icon: "🚀" },
  { label: "Send Email", skillName: "send_email", riskLevel: "medium", icon: "📧" },
  { label: "DB Query", skillName: "db_query", riskLevel: "medium", icon: "📊" },
  { label: "API Call", skillName: "api_call", riskLevel: "low", icon: "🔗" },
  { label: "File Write", skillName: "file_write", riskLevel: "medium", icon: "📝" },
  { label: "Approve Gate", skillName: "require_approval", riskLevel: "high", icon: "✅" },
];

/* ------------------------------------------------------------------ */
/* Main Editor Component                                               */
/* ------------------------------------------------------------------ */

export function WorkflowEditorPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [planName, setPlanName] = useState("Untitled Plan");
  const [planDesc, setPlanDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const nextId = useRef(0);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            animated: true,
            style: { stroke: "#00FFAA", strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: "#00FFAA" },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const addNode = (item: (typeof PALETTE_ITEMS)[0]) => {
    const id = `action-${nextId.current++}`;
    const newNode: Node = {
      id,
      type: "action",
      position: { x: 250 + Math.random() * 200, y: 100 + nextId.current * 100 },
      data: {
        label: item.label,
        skillName: item.skillName,
        description: "",
        riskLevel: item.riskLevel,
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const updateNodeData = (nodeId: string, field: string, value: string) => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, [field]: value } } : n
      )
    );
    if (selectedNode?.id === nodeId) {
      setSelectedNode((prev) =>
        prev ? { ...prev, data: { ...prev.data, [field]: value } } : prev
      );
    }
  };

  const deleteNode = (nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    if (selectedNode?.id === nodeId) setSelectedNode(null);
  };

  // Convert canvas to Plan API payload
  const savePlan = async () => {
    setSaving(true);
    setSaved(false);
    try {
      // Build node index map for dependency resolution
      const nodeIndexMap: Record<string, number> = {};
      nodes.forEach((n, i) => { nodeIndexMap[n.id] = i; });

      // Build dependency map from edges
      const depsMap: Record<string, number[]> = {};
      edges.forEach((e) => {
        if (!depsMap[e.target]) depsMap[e.target] = [];
        const sourceIdx = nodeIndexMap[e.source];
        if (sourceIdx !== undefined) depsMap[e.target].push(sourceIdx);
      });

      const actions = nodes.map((n, _i) => ({
        name: (n.data as ActionNodeData).label,
        skill_name: (n.data as ActionNodeData).skillName,
        description: (n.data as ActionNodeData).description || "",
        dependencies: depsMap[n.id] || [],
      }));

      await plans.create({ name: planName, description: planDesc, actions });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-0px)]">
      <HudTopBar
        title="Workflow Editor"
        subtitle={`${nodes.length} actions, ${edges.length} connections`}
        actions={
          <div className="flex items-center gap-2">
            {saved && <span className="text-xs text-hud-success">Saved!</span>}
            <button
              className="hud-btn hud-btn--primary text-xs"
              onClick={savePlan}
              disabled={saving || nodes.length === 0}
            >
              {saving ? "Saving..." : "Save as Plan"}
            </button>
          </div>
        }
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Left palette */}
        <div className="w-52 shrink-0 border-r border-white/[0.06] bg-hud-bg-dark p-3 overflow-y-auto hud-scroll">
          <div className="text-[10px] text-white/30 uppercase tracking-wider mb-2 px-1">Actions</div>
          <div className="space-y-1">
            {PALETTE_ITEMS.map((item) => (
              <button
                key={item.skillName}
                onClick={() => addNode(item)}
                className="w-full text-left rounded-lg border border-white/[0.06] hover:border-hud-accent/30 bg-white/[0.02] hover:bg-white/[0.04] px-3 py-2 transition-all"
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm">{item.icon}</span>
                  <div>
                    <div className="text-xs text-white/80">{item.label}</div>
                    <div className="text-[9px] font-mono text-white/25">{item.skillName}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>

          <div className="text-[10px] text-white/30 uppercase tracking-wider mt-4 mb-2 px-1">Plan Info</div>
          <input
            className="hud-input text-xs mb-2"
            value={planName}
            onChange={(e) => setPlanName(e.target.value)}
            placeholder="Plan name"
          />
          <textarea
            className="hud-input text-xs h-16 resize-none"
            value={planDesc}
            onChange={(e) => setPlanDesc(e.target.value)}
            placeholder="Description..."
          />
        </div>

        {/* Canvas */}
        <div className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: "#00FFAA44", strokeWidth: 2 },
              markerEnd: { type: MarkerType.ArrowClosed, color: "#00FFAA" },
            }}
            style={{ background: "#070A0F" }}
          >
            <Background color="#1A2540" gap={20} size={1} />
            <Controls
              className="!bg-hud-bg-medium !border-white/10 !rounded-lg [&>button]:!bg-hud-bg-medium [&>button]:!border-white/10 [&>button]:!text-white/60"
            />
            <MiniMap
              nodeColor="#00BFA5"
              maskColor="rgba(7, 10, 15, 0.8)"
              className="!bg-hud-bg-dark !border-white/10 !rounded-lg"
            />
          </ReactFlow>
        </div>

        {/* Right properties panel */}
        {selectedNode && (
          <div className="w-64 shrink-0 border-l border-white/[0.06] bg-hud-bg-dark p-4 overflow-y-auto hud-scroll">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs text-white/40 uppercase tracking-wider">Properties</div>
              <button
                className="text-[10px] text-hud-error/60 hover:text-hud-error"
                onClick={() => deleteNode(selectedNode.id)}
              >
                Delete
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] text-white/30 uppercase">Name</label>
                <input
                  className="hud-input text-xs mt-1"
                  value={(selectedNode.data as ActionNodeData).label}
                  onChange={(e) => updateNodeData(selectedNode.id, "label", e.target.value)}
                />
              </div>
              <div>
                <label className="text-[10px] text-white/30 uppercase">Skill</label>
                <input
                  className="hud-input text-xs mt-1 font-mono"
                  value={(selectedNode.data as ActionNodeData).skillName}
                  onChange={(e) => updateNodeData(selectedNode.id, "skillName", e.target.value)}
                />
              </div>
              <div>
                <label className="text-[10px] text-white/30 uppercase">Description</label>
                <textarea
                  className="hud-input text-xs mt-1 h-16 resize-none"
                  value={(selectedNode.data as ActionNodeData).description}
                  onChange={(e) => updateNodeData(selectedNode.id, "description", e.target.value)}
                />
              </div>
              <div>
                <label className="text-[10px] text-white/30 uppercase">Risk Level</label>
                <select
                  className="hud-select text-xs mt-1"
                  value={(selectedNode.data as ActionNodeData).riskLevel}
                  onChange={(e) => updateNodeData(selectedNode.id, "riskLevel", e.target.value)}
                >
                  <option value="safe">Safe (0-20)</option>
                  <option value="low">Low (20-40)</option>
                  <option value="medium">Medium (40-60)</option>
                  <option value="high">High (60-80)</option>
                  <option value="critical">Critical (80-100)</option>
                </select>
              </div>
              <div className="pt-2 border-t border-white/5 text-[10px] text-white/15 font-mono">
                ID: {selectedNode.id}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
