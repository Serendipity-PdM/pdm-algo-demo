import { useEffect, useState } from "react";

export default function ShiftRiskDashboard() {
  const [shiftData, setShiftData] = useState([]);
  const [filter, setFilter] = useState("All");
  const filteredData =
  filter === "All"
    ? shiftData
    : shiftData.filter((entry) => entry.risk_factor === filter);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    shift_type: "Morning",
    operator_id: 1,
    experience_level: "Intermediate",
    age: 30,
    gender: "Male",
    avg_week_hours: 45,
    last_year_incidents: 1,
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/load_shift_data")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setShiftData(data);
        } else {
          console.error("Invalid data format:", data);
          setShiftData([]);
        }
      })
      .catch((err) => {
        console.error("Failed to load shift data", err);
        setShiftData([]); // Avoid crash on render
      });
  }, []);  


  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]:
        ["operator_id", "age", "avg_week_hours", "last_year_incidents"].includes(
          name
        )
          ? Number(value)
          : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const response = await fetch("/append_shift_entry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

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
    setLoading(false);
  };

  return (
    <div className="p-6 max-w-screen-xl mx-auto">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Shift Risk Dashboard</h2>
        <button
          onClick={() => setShowForm((prev) => !prev)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {showForm ? "Hide Form" : "Add New Entry"}
        </button>
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
              <th className="p-2 text-left border">Worker No</th>
              {[
                "shift_type",
                "operator_id",
                "experience_level",
                "age",
                "gender",
                "avg_week_hours",
                "last_year_incidents",
                "predicted_time_cycles",
                "risk_factor",
              ].map((col) => (
                <th key={col} className="p-2 text-left capitalize border">
                  {col.replaceAll("_", " ")}
                </th>
              ))}
            </tr>
          </thead>
                    <tbody>
            {Array.isArray(filteredData) && filteredData.map((row, idx) => (
              <tr key={idx} className="odd:bg-white even:bg-gray-50">
                <td className="p-2 border font-semibold">{idx + 1}</td>
                {[
                  "shift_type",
                  "operator_id",
                  "experience_level",
                  "age",
                  "gender",
                  "avg_week_hours",
                  "last_year_incidents",
                  "predicted_time_cycles",
                  "risk_factor",
                ].map((key) => (
                  <td key={key} className="p-2 border">
                    {row[key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
