import { useState } from "react";

export default function ShiftPredictionForm() {
  const [formData, setFormData] = useState({
    shift_type: "Morning",
    operator_id: 1,
    experience_level: "Intermediate",
    age: 30,
    gender: "Male",
    avg_week_hours: 45,
    last_year_incidents: 1,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === "operator_id" || name === "age" || name === "avg_week_hours" || name === "last_year_incidents"
        ? Number(value)
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
  
    const numeric = [
      formData.operator_id,
      formData.age,
      formData.avg_week_hours,
      formData.last_year_incidents,
    ];
  
    const shiftMap = {
      Morning:    [1, 0, 0],
      Afternoon:  [0, 1, 0],
      Night:      [0, 0, 1],
    };
    
    const expMap = {
      Intern:        [1, 0, 0, 0, 0],
      Beginner:      [0, 1, 0, 0, 0],
      Intermediate:  [0, 0, 1, 0, 0],
      Experienced:   [0, 0, 0, 1, 0],
      Expert:        [0, 0, 0, 0, 1],
    };
    
    const genderMap = {
      Male:   [1, 0],
      Female: [0, 1],
    };
  
    const dummies = [
      ...shiftMap[formData.shift_type],        
      ...expMap[formData.experience_level],    
      ...genderMap[formData.gender],           
    ];
  
    const extraFeature = 0; 
    const vector = [
    ...numeric,
    ...shiftMap[formData.shift_type],
    ...expMap[formData.experience_level],
    ...genderMap[formData.gender],
       extraFeature 
];  
  
    if (vector.length !== 15) {
      console.error("Invalid vector length:", vector.length, vector);
      setResult("Feature vector incorrect.");
      setLoading(false);
      return;
    }
  
    try {
      const response = await fetch("http://localhost:8000/predict_shift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vector }),
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error response:", errorText);
        setResult("Error: " + response.status);
        return;
      }
  
      const data = await response.json();
      console.log("Prediction response:", data);
  
      if (data && typeof data.predicted_time_cycles === "number") {
        setResult(data.predicted_time_cycles.toFixed(2));
      } else {
        console.error("Unexpected response format:", data);
        setResult("Invalid response");
      }
    } catch (error) {
      console.error("Prediction failed:", error);
      setResult("Network error");
    } finally {
      setLoading(false);
    }
  };
  
  
  

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-xl font-bold">Predict Risk Level from Shift Data</h2>
      <form onSubmit={handleSubmit} className="space-y-3">

        <label className="block">
          Shift Type:
          <select name="shift_type" value={formData.shift_type} onChange={handleChange} className="ml-2">
            <option>Morning</option>
            <option>Afternoon</option>
            <option>Night</option>
          </select>
        </label>

        <label className="block">
          Operator ID:
          <input type="number" name="operator_id" value={formData.operator_id} onChange={handleChange} className="ml-2" />
        </label>

        <label className="block">
          Experience Level:
          <select name="experience_level" value={formData.experience_level} onChange={handleChange} className="ml-2">
            <option>Intern</option>
            <option>Beginner</option>
            <option>Intermediate</option>
            <option>Experienced</option>
            <option>Expert</option>
          </select>
        </label>

        <label className="block">
          Age:
          <input type="number" name="age" value={formData.age} onChange={handleChange} className="ml-2" />
        </label>

        <label className="block">
          Gender:
          <select name="gender" value={formData.gender} onChange={handleChange} className="ml-2">
            <option>Male</option>
            <option>Female</option>
          </select>
        </label>

        <label className="block">
          Avg Weekly Hours:
          <input type="number" name="avg_week_hours" value={formData.avg_week_hours} onChange={handleChange} className="ml-2" />
        </label>

        <label className="block">
          Last Year Incidents:
          <input type="number" name="last_year_incidents" value={formData.last_year_incidents} onChange={handleChange} className="ml-2" />
        </label>

        <button type="submit" className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {result !== null && (
        <div className="mt-4">
          <strong>Predicted Risk Number:</strong> {result}
        </div>
      )}
    </div>
  );
}
