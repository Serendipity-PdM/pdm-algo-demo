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
    try {
      const params = new URLSearchParams({
        offset: startOffset,
        limit: 500,
      });

      if (sort.key && sort.direction) {
        params.append("sort_key", sort.key);
        params.append("sort_direction", sort.direction);
      }

      const res = await fetch(`http://localhost:8000/load_shift_data?${params.toString()}`);
      const data = await res.json();

      if (Array.isArray(data)) {
        if (startOffset === 0) {
          setShiftData(data);
        } else {
          setShiftData((prev) => [...prev, ...data]);
        }

        setOffset(startOffset + 500);
        setHasMore(data.length === 500);

        setShowLoadMore(false);
        setTimeout(() => {
          setShowLoadMore(true);
        }, 1500);
      } else {
        console.error("Invalid data format:", data);
      }
    } catch (err) {
      console.error("Failed to load shift data", err);
    }
  };

  const loadAllData = async () => {
    setLoadingAll(true);
    setLoadingDots(".");
    let fullData = [];
    let pageOffset = 0;
    const pageLimit = 1000;

    while (true) {
      const res = await fetch(`http://localhost:8000/load_shift_data?offset=${pageOffset}&limit=${pageLimit}`);
      const batch = await res.json();

      if (!Array.isArray(batch) || batch.length === 0) break;

      fullData = [...fullData, ...batch];
      pageOffset += pageLimit;

      if (batch.length < pageLimit) break;
    }

    setShiftData(fullData);
    setOffset(pageOffset);
    setHasMore(false);
    setLoadingAll(false);
  };

  const sortedData = [...filteredData].sort((a, b) => {
    const { key, direction } = sortConfig;
    if (!key || !direction) return 0;

    const aVal = a[key];
    const bVal = b[key];

    if (aVal == null) return 1;
    if (bVal == null) return -1;

    if (typeof aVal === "number" && typeof bVal === "number") {
      return direction === "asc" ? aVal - bVal : bVal - aVal;
    }

    return direction === "asc"
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
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

    const shiftMap = {
      Morning: [1, 0, 0],
      Afternoon: [0, 1, 0],
      Night: [0, 0, 1],
    };

    const expMap = {
      Intern: [1, 0, 0, 0, 0],
      Beginner: [0, 1, 0, 0, 0],
      Intermediate: [0, 0, 1, 0, 0],
      Experienced: [0, 0, 0, 1, 0],
      Expert: [0, 0, 0, 0, 1],
    };

    const genderMap = {
      Male: [1, 0],
      Female: [0, 1],
    };

    const extraFeature = 0;

    const vector = [
      ...numeric,
      ...shiftMap[formData.shift_type],
      ...expMap[formData.experience_level],
      ...genderMap[formData.gender],
      extraFeature,
    ];

    if (vector.length !== 15) {
      console.error("Invalid vector length:", vector.length, vector);
      setLoading(false);
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/append_shift_entry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vector }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error response:", errorText);
        setLoading(false);
        return;
      }

      const newEntry = await response.json();
      setShiftData((prev) => [...prev, newEntry]);
      setFormData({
        shift_type: "Morning",
        operator_id: 1,
        experience_level: "Intermediate",
        age: 30,
        gender: "Male",
        avg_week_hours: 45,
        last_year_incidents: 1,
      });
      setShowForm(false);
    } catch (err) {
      console.error("Failed to submit entry:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-screen-xl mx-auto">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Shift Risk Dashboard</h2>
        <div className="flex gap-2">
          <button
            onClick={loadAllData}
            disabled={loadingAll}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
          >
            {loadingAll ? `Loading${loadingDots}` : "Load All Board"}
          </button>
          <button
            onClick={() => setShowForm((prev) => !prev)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            {showForm ? "Hide Form" : "Add New Entry"}
          </button>
        </div>
      </div>

      <div className="mb-4">
        <label className="mr-2 font-medium">Filter by Risk:</label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="border rounded px-2 py-1"
        >
          <option>All</option>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
        </select>
      </div>

      {showForm && (
        <form
          onSubmit={handleSubmit}
          className="grid grid-cols-2 gap-4 bg-gray-100 p-4 rounded mb-6"
        >
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
                className="mt-1 p-2 border rounded"
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
                className="mt-1 p-2 border rounded"
              />
            </label>
          ))}

          <button
            type="submit"
            disabled={loading}
            className="col-span-2 mt-4 bg-green-600 text-white py-2 rounded hover:bg-green-700"
          >
            {loading ? "Submitting..." : "Submit Entry & Predict"}
          </button>
        </form>
      )}

      <div className="overflow-x-auto">
        <table className="w-full border text-sm">
        <thead className="bg-gray-200">
          <tr>
            {["Worker No",
              "shift_type",
              "operator_id",
              "experience_level",
              "age",
              "gender",
              "avg_week_hours",
              "last_year_incidents",
              "predicted_time_cycles",
              "risk_factor"
            ].map((col) => {
              const isSortable = ["Worker No", "operator_id", "age", "avg_week_hours", "last_year_incidents", "predicted_time_cycles"].includes(col);
              const key = col === "Worker No" ? "index" : col;

              return (
                <th
                  key={col}
                  onClick={() => {
                    if (!isSortable) return;
                    setSortConfig((prev) => {
                      if (prev.key === key && prev.direction === "asc") {
                        return { key, direction: "desc" };
                      } else if (prev.key === key && prev.direction === "desc") {
                        return { key: null, direction: null }; 
                      } else {
                        return { key, direction: "asc" };
                      }
                    });
                  }}
                  className={`p-2 text-left border capitalize cursor-pointer select-none ${
                    sortConfig.key === key ? "bg-gray-300" : ""
                  }`}
                >
                  {col}
                  {sortConfig.key === key && (
                    <span className="ml-1 text-xs">
                      {sortConfig.direction === "asc" ? "▲" : sortConfig.direction === "desc" ? "▼" : ""}
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
              .map((row, idx) => ({ ...row, index: idx + 1 })) // Add Worker No
              .sort((a, b) => {
                const { key, direction } = sortConfig;
                if (!key || !direction) return 0;
                if (key === "index") return direction === "asc" ? a.index - b.index : b.index - a.index;
                return direction === "asc"
                  ? Number(a[key]) - Number(b[key])
                  : Number(b[key]) - Number(a[key]);
              })
              .map((row, idx) => (
                <tr key={idx} className="odd:bg-white even:bg-gray-50">
                  {["index", "shift_type", "operator_id", "experience_level", "age", "gender", "avg_week_hours", "last_year_incidents", "predicted_time_cycles", "risk_factor"].map((key) => (
                    <td
                      key={key}
                      className={`p-2 border ${
                        sortConfig.key === key ? "bg-gray-100" : ""
                      }`}
                    >
                      {key === "risk_factor" ? (
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded-full ${
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
            className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-900"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  );
}
