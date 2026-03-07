export type LockedLineup = {
  lineup: string;
  min: string;
  plusMinus: string;
  net: string;
  ortg: string;
  drtg: string;
  offPoss: number;
  defPoss: number;
  score: string;
};

export const q4GroundTruth = {
  on: { min: "11:36", offPoss: 26, defPoss: 24, score: "HOU 29-15 POR" },
  off: { min: "0:24", offPoss: 0, defPoss: 2, score: "HOU 0-2 POR" }
};

export const q4OnLocks: LockedLineup[] = [
  { lineup: "A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela", min: "4:46", plusMinus: "4", net: "20.2", ortg: "109.1", drtg: "88.9", offPoss: 11, defPoss: 9, score: "HOU 12-8 POR" },
  { lineup: "A.Thompson, A.Sengun, R.Sheppard, D.Finney-Smith, T.Eason", min: "3:31", plusMinus: "1", net: "12.5", ortg: "100", drtg: "87.5", offPoss: 8, defPoss: 8, score: "HOU 8-7 POR" },
  { lineup: "R.Sheppard, K.Durant, T.Eason, J.Okogie, C.Capela", min: "2:00", plusMinus: "2", net: "50", ortg: "50", drtg: "0", offPoss: 4, defPoss: 4, score: "HOU 2-0 POR" },
  { lineup: "A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela", min: "1:19", plusMinus: "7", net: "233.3", ortg: "233.3", drtg: "0", offPoss: 3, defPoss: 3, score: "HOU 7-0 POR" }
];

export const q4OffLock: LockedLineup = {
  lineup: "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela",
  min: "0:24",
  plusMinus: "-2",
  net: "-100",
  ortg: "0",
  drtg: "100",
  offPoss: 0,
  defPoss: 2,
  score: "HOU 0-2 POR"
};

export const fullGameGroundTruth = {
  on: { min: "35:40", offPoss: 75, defPoss: 72, ortg: "110.7", drtg: "105.6", net: "5.1" },
  off: { min: "12:20", offPoss: 24, defPoss: 26, ortg: "95.8", drtg: "88.5", net: "7.4" }
};
