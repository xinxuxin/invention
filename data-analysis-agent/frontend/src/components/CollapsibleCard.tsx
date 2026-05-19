import { useEffect, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

import { cn } from "../lib/utils";

type CollapsibleCardProps = {
  title: string;
  eyebrow?: string;
  icon?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  actions?: ReactNode;
};

export function CollapsibleCard({
  title,
  eyebrow,
  icon,
  defaultOpen = true,
  children,
  actions,
}: CollapsibleCardProps) {
  const [open, setOpen] = useState(defaultOpen);

  useEffect(() => {
    if (defaultOpen) {
      setOpen(true);
    }
  }, [defaultOpen]);

  return (
    <section className="overflow-hidden rounded-lg border border-border bg-white/74">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-white/70"
      >
        <span className="flex min-w-0 items-center gap-3">
          {icon ? (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-slate-950 text-white">
              {icon}
            </span>
          ) : null}
          <span className="min-w-0">
            {eyebrow ? (
              <span className="block text-[11px] font-bold uppercase tracking-[0.16em] text-muted-foreground">
                {eyebrow}
              </span>
            ) : null}
            <span className="block truncate text-sm font-bold">{title}</span>
          </span>
        </span>
        <span className="flex items-center gap-2">
          {actions}
          <ChevronDown
            className={cn("h-4 w-4 text-muted-foreground transition", open ? "rotate-180" : "")}
          />
        </span>
      </button>
      {open ? <div className="border-t border-border/70 p-4">{children}</div> : null}
    </section>
  );
}
