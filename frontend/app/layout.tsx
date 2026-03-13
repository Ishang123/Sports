import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Prediction Market Integrity Dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <div className="banner">
            Anomaly detection for integrity research. Scores are not proof of wrongdoing.
          </div>
          {children}
        </div>
      </body>
    </html>
  );
}
