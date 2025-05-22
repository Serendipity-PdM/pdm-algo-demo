import { useEffect, useState } from "react";

export default function ShiftRiskDashboard() {
  const [shiftData, setShiftData] = useState([]);
  const [filter, setFilter] = useState("All");
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showLoadMore, setShowLoadMore] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: "asc" });
  const [loadingAll, setLoadingAll] = useState(false);
  const [loadingDots, setLoadingDots] = useState(".");
  const [newlyAddedIndex, setNewlyAddedIndex] = useState(null);
  const [formData, setFormData] = useState({
    shift_type: "Morning",
    operator_id: 1,
    experience_level: "Intermediate",
    age: 30,
    gender: "Male",
    avg_week_hours: 45,
    last_year_incidents: 1,
  });

  const filteredData =
    filter === "All"
      ? shiftData
      : shiftData.filter((entry) => entry.risk_factor === filter);

  useEffect(() => {
    loadShiftData(0);
  }, []);

  useEffect(() => {
    if (!loadingAll) return;
    const interval = setInterval(() => {
      setLoadingDots((prev) => (prev === "..." ? "." : prev + "."));
    }, 400);
    return () => clearInterval(interval);
  }, [loadingAll]);

  const loadShiftData = async (startOffset, sort = sortConfig) => {
    const params = new URLSearchParams({ offset: startOffset, limit: 500 });
    if (sort.key && sort.direction) {
      params.append("sort_key", sort.key);
      params.append("sort_direction", sort.direction);
    }

    const res = await fetch(`http://localhost:8000/load_shift_data?${params.toString()}`);
    const data = await res.json();

    if (Array.isArray(data)) {
      if (startOffset === 0) setShiftData(data);
      else setShiftData((prev) => [...prev, ...data]);

      setOffset(startOffset + 500);
      setHasMore(data.length === 500);
      setShowLoadMore(false);
      setTimeout(() => setShowLoadMore(true), 1500);
    }
  };

  const loadAllData = async () => {
    setLoadingAll(true);
    setLoadingDots(".");
    let fullData = [];
    let pageOffset = 0;

    while (true) {
      const res = await fetch(`http://localhost:8000/load_shift_data?offset=${pageOffset}&limit=1000`);
      const batch = await res.json();
      if (!Array.isArray(batch) || batch.length === 0) break;
      fullData = [...fullData, ...batch];
      pageOffset += 1000;
      if (batch.length < 1000) break;
    }

    setShiftData(fullData);
    setOffset(pageOffset);
    setHasMore(false);
    setLoadingAll(false);
  };

  const sortedData = [...filteredData].sort((a, b) => {
    const { key, direction } = sortConfig;
    if (!key || !direction) return 0;
    const aVal = a[key], bVal = b[key];
    if (aVal == null) return 1;
    if (bVal == null) return -1;
    return typeof aVal === "number"
      ? (direction === "asc" ? aVal - bVal : bVal - aVal)
      : (direction === "asc"
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal)));
  });

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: ["operator_id", "age", "avg_week_hours", "last_year_incidents"].includes(name)
        ? Number(value)
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const numeric = [
      formData.operator_id,
      formData.age,
      formData.avg_week_hours,
      formData.last_year_incidents,
    ];
    const shiftMap = { Morning: [1, 0, 0], Afternoon: [0, 1, 0], Night: [0, 0, 1] };
    const expMap = {
      Intern: [1, 0, 0, 0, 0], Beginner: [0, 1, 0, 0, 0],
      Intermediate: [0, 0, 1, 0, 0], Experienced: [0, 0, 0, 1, 0], Expert: [0, 0, 0, 0, 1],
    };
    const genderMap = { Male: [1, 0], Female: [0, 1] };
    const vector = [...numeric, ...shiftMap[formData.shift_type], ...expMap[formData.experience_level], ...genderMap[formData.gender], 0];

    if (vector.length !== 15) return setLoading(false);

    try {
      const response = await fetch("http://localhost:8000/append_shift_entry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vector }),
      });
      const newEntry = await response.json();
      setShiftData((prev) => [...prev, newEntry]);
      setNewlyAddedIndex(prev.length); // for animation
      setTimeout(() => setNewlyAddedIndex(null), 2000);

      setFormData({
        shift_type: "Morning", operator_id: 1, experience_level: "Intermediate",
        age: 30, gender: "Male", avg_week_hours: 45, last_year_incidents: 1,
      });
      setShowForm(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-screen-xl mx-auto bg-white rounded-lg shadow-md relative">
      <style>
        {`
          .modal-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0,0,0,0.4); display: flex;
            align-items: center; justify-content: center; z-index: 50;
          }
          .modal-content {
            background: white; padding: 2rem; border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); width: 100%;
            max-width: 600px;
          }
          .flash {
            animation: flash-bg 1s ease-in-out;
          }
          @keyframes flash-bg {
            0% { background-color: #e6fffa; }
            100% { background-color: transparent; }
          }
        `}
      </style>

      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-extrabold text-gray-800">Shift Risk Dashboard</h2>
        <div className="flex gap-3">
          <button
            onClick={loadAllData}
            disabled={loadingAll}
            className={`px-4 py-2 rounded font-medium text-white ${
              loadingAll ? "bg-purple-400" : "bg-purple-600 hover:bg-purple-700"
            }`}
          >
            {loadingAll ? `Loading${loadingDots}` : "Load All Board"}
          </button>
          <button
            onClick={() => setShowForm(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium"
          >
            Add New Entry
          </button>
        </div>
      </div>

      <div className="mb-5">
        <label className="mr-3 font-semibold text-gray-700">Filter by Risk:</label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 rounded-full border border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-semibold text-gray-800 bg-white transition"
        >
          <option className="text-gray-700">All</option>
          <option className="text-green-700 bg-green-100">Low</option>
          <option className="text-yellow-800 bg-yellow-100">Medium</option>
          <option className="text-red-700 bg-red-100">High</option>
        </select>
      </div>


      {showForm && (
        <div className="modal-overlay">
          <div className="modal-content">
            <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
              <div className="col-span-2 flex justify-between items-center mb-2">
                <h3 className="text-lg font-bold">Add New Shift Entry</h3>
                <button type="button" onClick={() => setShowForm(false)} className="px-4 py-2 rounded-md font-semibold shadow-sm transition-colors duration-200 bg-blue-600 text-white hover:bg-blue-700">✕</button>
              </div>
              {[
                ["shift_type", "Shift Type", ["Morning", "Afternoon", "Night"]],
                ["experience_level", "Experience Level", ["Intern", "Beginner", "Intermediate", "Experienced", "Expert"]],
                ["gender", "Gender", ["Male", "Female"]],
              ].map(([name, label, options]) => (
                <label key={name} className="flex flex-col">
                  {label}
                  <select
                    name={name}
                    value={formData[name]}
                    onChange={handleFormChange}
                    className="mt-1 p-2 border border-gray-300 rounded"
                  >
                    {options.map((opt) => (
                      <option key={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
              ))}
              {[
                ["operator_id", "Operator ID"],
                ["age", "Age"],
                ["avg_week_hours", "Avg Weekly Hours"],
                ["last_year_incidents", "Incidents Last Year"],
              ].map(([name, label]) => (
                <label key={name} className="flex flex-col">
                  {label}
                  <input
                    type="number"
                    name={name}
                    value={formData[name]}
                    onChange={handleFormChange}
                    className="mt-1 p-2 border border-gray-300 rounded"
                    min={name === "operator_id" ? 1 : name === "age" ? 20 : name === "avg_week_hours" ? 30 : 0}
                    max={name === "operator_id" ? 10 : name === "age" ? 65 : name === "avg_week_hours" ? 70 : 15}
                  />
                </label>
              ))}
              <button
                type="submit"
                disabled={loading}
                className="col-span-2 mt-3 bg-green-600 text-white py-2 rounded hover:bg-green-700"
              >
                {loading ? "Submitting..." : "Submit Entry & Predict"}
              </button>
            </form>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm text-gray-700 shadow-md">
        <thead className="sticky top-0 z-10 bg-blue-100 text-gray-900 font-semibold shadow-sm">
          <tr>
            {[
              "Worker No", "shift_type", "operator_id", "experience_level",
              "age", "gender", "avg_week_hours", "last_year_incidents",
              "predicted_time_cycles", "risk_factor"
            ].map((col) => {
              const isSortable = ["Worker No", "operator_id", "age", "avg_week_hours", "last_year_incidents", "predicted_time_cycles"].includes(col);
              const key = col === "Worker No" ? "index" : col;

              return (
                <th
                  key={col}
                  onClick={() => {
                    if (!isSortable) return;
                    setSortConfig((prev) =>
                      prev.key === key && prev.direction === "asc"
                        ? { key, direction: "desc" }
                        : prev.key === key && prev.direction === "desc"
                        ? { key: null, direction: null }
                        : { key, direction: "asc" }
                    );
                  }}
                  className={`p-2 border capitalize select-none text-left transition ${
                    isSortable ? "cursor-pointer hover:bg-gray-200" : ""
                  } ${sortConfig.key === key ? "bg-gray-300" : ""}`}
                >
                  {col.replace(/_/g, " ")}
                  {sortConfig.key === key && (
                    <span className="ml-1 text-xs">
                      {sortConfig.direction === "asc" ? "▲" : "▼"}
                    </span>
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
          <tbody>
            {Array.isArray(filteredData) &&
              filteredData
                .map((row, idx) => ({ ...row, index: idx + 1 }))
                .sort((a, b) => {
                  const { key, direction } = sortConfig;
                  if (!key || !direction) return 0;
                  return direction === "asc"
                    ? Number(a[key]) - Number(b[key])
                    : Number(b[key]) - Number(a[key]);
                })
                .map((row, idx) => (
                  <tr
                    key={idx}
                    className={`odd:bg-white even:bg-gray-50 hover:bg-green-100 transition-colors ${idx === newlyAddedIndex ? "flash" : ""}`}
                  >
                    {["index", "shift_type", "operator_id", "experience_level", "age", "gender", "avg_week_hours", "last_year_incidents", "predicted_time_cycles", "risk_factor"]
                      .map((key) => (
                        <td
                          key={key}
                          className={`p-2 border ${sortConfig.key === key ? "bg-gray-100" : ""}`}
                        >
                          {key === "risk_factor" ? (
                            <span
                              className={`px-3 py-1 text-xs font-semibold rounded-full shadow-sm ${
                                row[key] === "Low"
                                  ? "bg-green-100 text-green-700"
                                  : row[key] === "Medium"
                                  ? "bg-yellow-100 text-yellow-800"
                                  : row[key] === "High"
                                  ? "bg-red-100 text-red-700"
                                  : "bg-gray-100 text-gray-600"
                              }`}
                            >
                              {row[key]}
                            </span>
                          ) : (
                            row[key]
                          )}
                        </td>
                      ))}
                  </tr>
                ))}
          </tbody>
        </table>
      </div>

      {hasMore && showLoadMore && (
        <div className="mt-4 text-center">
          <button
            onClick={() => loadShiftData(offset)}
            className="px-4 py-2 rounded-md font-semibold shadow-sm transition-colors duration-200 bg-blue-600 text-white hover:bg-blue-700"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  );
}
