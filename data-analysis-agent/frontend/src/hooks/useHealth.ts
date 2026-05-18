import { useEffect, useState } from "react";

import { getHealth } from "../lib/api";
import type { HealthResponse } from "../types/api";

type HealthState =
  | { status: "loading"; data: null; error: null }
  | { status: "online"; data: HealthResponse; error: null }
  | { status: "offline"; data: null; error: string };

export function useHealth() {
  const [state, setState] = useState<HealthState>({ status: "loading", data: null, error: null });

  useEffect(() => {
    let isMounted = true;

    getHealth()
      .then((data) => {
        if (isMounted) {
          setState({ status: "online", data, error: null });
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
