import { useEffect, useState } from "react";

import { getHealth, getHealthConfig } from "../lib/api";
import type { HealthConfigResponse, HealthResponse } from "../types/api";

type HealthState =
  | { status: "loading"; data: null; error: null }
  | { status: "online"; data: HealthResponse; config: HealthConfigResponse | null; error: null }
  | { status: "offline"; data: null; error: string };

export function useHealth() {
  const [state, setState] = useState<HealthState>({ status: "loading", data: null, error: null });

  useEffect(() => {
    let isMounted = true;

    Promise.all([getHealth(), getHealthConfig().catch(() => null)])
      .then(([data, config]) => {
        if (isMounted) {
          setState({ status: "online", data, config, error: null });
        }
      })
      .catch((error: unknown) => {
        if (isMounted) {
          setState({
            status: "offline",
            data: null,
            error: error instanceof Error ? error.message : "Unable to reach API",
          });
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return state;
}
