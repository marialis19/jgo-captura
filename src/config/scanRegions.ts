export type ScanRegion =
  | "hand"
  | "full-arm"
  | "forearm"
  | "finger"
  | "ankle";

export type ScanRegionConfig = {
  id: ScanRegion;
  label: string;
  requiresHand: boolean;
  requiresPose: boolean;
};

export const scanRegions: ScanRegionConfig[] = [
  {
    id: "hand",
    label: "Mano completa",
    requiresHand: true,
    requiresPose: false
  },
  {
    id: "full-arm",
    label: "Brazo completo",
    requiresHand: true,
    requiresPose: true
  },
  {
    id: "forearm",
    label: "Antebrazo completo",
    requiresHand: true,
    requiresPose: false
  },
  {
    id: "finger",
    label: "Dedo individual",
    requiresHand: true,
    requiresPose: false
  },
  {
    id: "ankle",
    label: "Tobillo",
    requiresHand: false,
    requiresPose: true
  }
];