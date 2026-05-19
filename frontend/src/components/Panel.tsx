import type { HTMLAttributes } from "react";

import { cn } from "../lib/utils";

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <section
      className={cn("rounded-lg border border-white/70 bg-white/72 shadow-glow backdrop-blur-xl", className)}
      {...props}
    />
  );
}
