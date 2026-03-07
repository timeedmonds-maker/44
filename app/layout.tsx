import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AUDIT-GRADE CONTROL PARSER V11",
  description: "Q4 resolved lineup windows first"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
