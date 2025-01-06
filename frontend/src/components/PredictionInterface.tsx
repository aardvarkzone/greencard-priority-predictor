"use client";

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Calendar, Clock, AlertCircle, TrendingUp, CalendarClock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PredictionResult {
  waitTime: number;
  confidence: number;
  historicalTrend: { date: string; movement: number }[];
  estimatedDate: string;
}

export default function PredictionInterface() {
  const [formData, setFormData] = useState({
    priorityDate: '',
    visaCategory: '',
    country: '',
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Prediction failed');
      
      const data = await response.json();
      setPrediction({
        ...data,
        historicalTrend: generateMockTrend(), // Remove this in production
        estimatedDate: calculateEstimatedDate(new Date(), data.waitTime)
      });
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Helper function to generate mock trend data (remove in production)
  const generateMockTrend = () => {
    return Array.from({ length: 12 }, (_, i) => ({
      date: `2024-${String(i + 1).padStart(2, '0')}`,
      movement: Math.floor(Math.random() * 30) + 10
    }));
  };

  const calculateEstimatedDate = (startDate: Date, months: number) => {
    const date = new Date(startDate);
    date.setMonth(date.getMonth() + months);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long' });
  };

  const isFormValid = () => {
    return formData.priorityDate && formData.visaCategory && formData.country;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 p-4">
      <Card className="border-t-4 border-t-green-500">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-6 h-6 text-green-500" />
            Green Card Priority Date Predictor
          </CardTitle>
          <CardDescription>
            Get estimated processing times based on historical visa bulletin data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <CalendarClock className="w-4 h-4 text-gray-500" />
                  Priority Date
                </label>
                <input
                  type="date"
                  className="w-full rounded-md border border-input bg-background px-3 py-2"
                  value={formData.priorityDate}
                  onChange={(e) => setFormData({...formData, priorityDate: e.target.value})}
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-gray-500" />
                  Visa Category
                </label>
                <Select onValueChange={(value) => setFormData({...formData, visaCategory: value})}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="EB1">EB1 - Priority Workers</SelectItem>
                    <SelectItem value="EB2">EB2 - Advanced Degrees</SelectItem>
                    <SelectItem value="EB3">EB3 - Skilled Workers</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-gray-500" />
                  Country of Chargeability
                </label>
                <Select onValueChange={(value) => setFormData({...formData, country: value})}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select country" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="india">India</SelectItem>
                    <SelectItem value="china">China</SelectItem>
                    <SelectItem value="philippines">Philippines</SelectItem>
                    <SelectItem value="other">Other Countries</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button 
              type="submit" 
              className="w-full bg-green-500 hover:bg-green-600" 
              disabled={loading || !isFormValid()}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Clock className="w-4 h-4 animate-spin" />
                  Calculating...
                </span>
              ) : 'Get Prediction'}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {prediction && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>Based on historical visa bulletin data analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="p-4 bg-green-50 rounded-lg">
                  <h3 className="font-semibold text-lg text-green-700">Estimated Wait Time</h3>
                  <p className="text-4xl font-bold text-green-700">
                    {prediction.waitTime} months
                  </p>
                  <p className="text-sm text-green-600 mt-2">
                    Estimated approval: {prediction.estimatedDate}
                  </p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence Level</span>
                    <span className="font-semibold">{prediction.confidence}%</span>
                  </div>
                  <Progress value={prediction.confidence} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Historical Movement Trend</CardTitle>
              <CardDescription>Priority date movement over the last 12 months</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prediction.historicalTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short' })}
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`${value} days`, 'Movement']}
                      labelFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="movement" 
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={{ fill: '#22c55e' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}