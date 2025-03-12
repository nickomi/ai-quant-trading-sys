import { useState, useEffect } from "react";

export default function TradingDashboard() {
  const [status, setStatus] = useState("");

  const executeTrade = async () => {
    const response = await fetch("http://localhost:8000/trade");
    const data = await response.json();
    setStatus(data.status);
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold">AI Trading Dashboard</h1>
      <button className="bg-blue-500 text-white px-4 py-2 rounded" onClick={executeTrade}>
        Execute Trade
      </button>
      <p>{status}</p>
    </div>
  );
}