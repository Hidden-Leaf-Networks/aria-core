interface HudToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export function HudToggle({ checked, onChange, label, disabled }: HudToggleProps) {
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer">
      <button
        role="switch"
        aria-checked={checked}
        data-state={checked ? "checked" : "unchecked"}
        disabled={disabled}
        className="hud-toggle"
        onClick={() => onChange(!checked)}
      >
        <span className="hud-toggle-thumb" />
      </button>
      {label && <span className="text-sm text-white/70">{label}</span>}
    </label>
  );
}
