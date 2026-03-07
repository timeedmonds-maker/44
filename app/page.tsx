import type { ReactNode } from "react";
import { fullGameGroundTruth, q4GroundTruth, q4OnLocks, q4OffLock } from "../lib/data";
import { getQ4ParserOutput } from "../lib/parser";

function Table({ headers, rows }: { headers: string[]; rows: ReactNode[][] }) {
  return (
    <table>
      <thead>
        <tr>{headers.map((h) => <th key={h}>{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export default function Page() {
  const parsed = getQ4ParserOutput();

  return (
    <main className="page">
      <h1>Rockets Live Game Dashboard</h1>
      <h2 className="eyebrow">AUDIT-GRADE CONTROL PARSER V11</h2>
      <p className="lede">
        Real resolved-Q4-window pass first. This version starts surfacing actual parser rows from <code>lib/parser.ts</code>,
        while keeping possession attribution deliberately staged behind the locked Q4 lineup rows.
      </p>

      <div className="chips">
        <span>Q4 only first</span>
        <span>resolved lineup windows</span>
        <span>parser vs screenshot locks</span>
        <span>first wrong possession</span>
        <span>no full-game widening yet</span>
      </div>

      <Section title="Locked Q4 ground truth">
        <Table
          headers={["Split", "MIN", "Off Poss", "Def Poss", "Score"]}
          rows={[
            ["ON", q4GroundTruth.on.min, q4GroundTruth.on.offPoss, q4GroundTruth.on.defPoss, q4GroundTruth.on.score],
            ["OFF", q4GroundTruth.off.min, q4GroundTruth.off.offPoss, q4GroundTruth.off.defPoss, q4GroundTruth.off.score]
          ]}
        />
      </Section>

      <Section title="Q4 Reed-on lineup locks">
        <Table
          headers={["Lineup", "MIN", "+/-", "Net", "ORtg", "DRtg", "Off Poss", "Def Poss", "Score"]}
          rows={q4OnLocks.map((r) => [r.lineup, r.min, r.plusMinus, r.net, r.ortg, r.drtg, r.offPoss, r.defPoss, r.score])}
        />
      </Section>

      <Section title="Q4 Reed-off lineup lock">
        <Table
          headers={["Lineup", "MIN", "+/-", "Net", "ORtg", "DRtg", "Off Poss", "Def Poss", "Score"]}
          rows={[[q4OffLock.lineup, q4OffLock.min, q4OffLock.plusMinus, q4OffLock.net, q4OffLock.ortg, q4OffLock.drtg, q4OffLock.offPoss, q4OffLock.defPoss, q4OffLock.score]]}
        />
      </Section>

      <Section title="Q4 resolved lineup windows">
        <p className="note">Now populated from parser output. Durations and merge reasons show where the next trim is required.</p>
        <Table
          headers={["Lineup", "Start", "End", "Duration", "Off Poss", "Def Poss", "HOU pts", "POR pts", "Source raw ids", "Merge reason"]}
          rows={parsed.resolvedWindows.map((w) => [w.lineup, w.start, w.end, w.duration, w.offPoss, w.defPoss, w.houPts, w.porPts, w.sourceRawIds, w.mergeReason])}
        />
      </Section>

      <Section title="Q4 parser vs locked lineup rows">
        <Table
          headers={["Lineup", "Parser MIN", "Target MIN", "Δ MIN", "Parser Off", "Target Off", "Δ Off", "Parser Def", "Target Def", "Δ Def", "Parser Score", "Target Score"]}
          rows={parsed.comparisonRows.map((r) => [r.lineup, r.parserMin, r.targetMin, r.deltaMin, r.parserOff, r.targetOff, r.deltaOff, r.parserDef, r.targetDef, r.deltaDef, r.parserScore, r.targetScore])}
        />
      </Section>

      <Section title="First wrong possession">
        {parsed.firstWrongPossession ? (
          <Table
            headers={["Possession id", "Period", "Start", "End", "Offense", "Trigger lineup", "Counted lineup", "Previous window", "Next window", "Why parser put it here"]}
            rows={[[
              parsed.firstWrongPossession.possessionId,
              parsed.firstWrongPossession.period,
              parsed.firstWrongPossession.start,
              parsed.firstWrongPossession.end,
              parsed.firstWrongPossession.offense,
              parsed.firstWrongPossession.triggerLineup,
              parsed.firstWrongPossession.countedLineup,
              parsed.firstWrongPossession.previousWindow,
              parsed.firstWrongPossession.nextWindow,
              parsed.firstWrongPossession.why
            ]]}
          />
        ) : (
          <p className="note">No divergence found.</p>
        )}
      </Section>

      <Section title="Implementation rules">
        <ul>
          <li>Admin-only clusters never create possessions or split them by themselves.</li>
          <li>Free throws inherit the trigger lineup, not the post-sub lineup.</li>
          <li>Defensive rebounds close possessions but do not create standalone zero-point rows.</li>
          <li>Micro-windows under 24 seconds are merged unless benchmark evidence proves they stand alone.</li>
          <li>Q4 lineup rows are the source of truth before full-game Reed aggregates.</li>
        </ul>
      </Section>

      <Section title="Locked full-game ground truth">
        <Table
          headers={["Split", "MIN", "Off Poss", "Def Poss", "ORtg", "DRtg", "Net"]}
          rows={[
            ["ON", fullGameGroundTruth.on.min, fullGameGroundTruth.on.offPoss, fullGameGroundTruth.on.defPoss, fullGameGroundTruth.on.ortg, fullGameGroundTruth.on.drtg, fullGameGroundTruth.on.net],
            ["OFF", fullGameGroundTruth.off.min, fullGameGroundTruth.off.offPoss, fullGameGroundTruth.off.defPoss, fullGameGroundTruth.off.ortg, fullGameGroundTruth.off.drtg, fullGameGroundTruth.off.net]
          ]}
        />
      </Section>
    </main>
  );
}
