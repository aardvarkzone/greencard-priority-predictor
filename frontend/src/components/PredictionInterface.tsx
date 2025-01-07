"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Calendar, Clock, TrendingUp, CalendarClock, Globe, Users, Briefcase, Moon, Sun } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

type VisaCategory = 'EB1' | 'EB2' | 'EB3' | 'F1' | 'F2A' | 'F2B' | 'F3' | 'F4';

interface PredictionResult {
  waitTime: number;
  confidence: number;
  historicalTrend: { date: string; movement: number }[];
  estimatedDate: string;
}

const CATEGORY_DESCRIPTIONS: Record<VisaCategory, string> = {
  'EB1': 'Priority Workers (Extraordinary Ability, Outstanding Researchers, Multinational Executives)',
  'EB2': 'Advanced Degree or Exceptional Ability',
  'EB3': 'Skilled Workers, Professionals, Other Workers',
  'F1': 'Unmarried Sons and Daughters of U.S. Citizens',
  'F2A': 'Spouses and Children of Permanent Residents',
  'F2B': 'Unmarried Sons and Daughters of Permanent Residents',
  'F3': 'Married Sons and Daughters of U.S. Citizens',
  'F4': 'Brothers and Sisters of Adult U.S. Citizens'
};

export default function PredictionInterface() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [activeCategory, setActiveCategory] = useState<'employment' | 'family'>('employment');
  const [formData, setFormData] = useState({
    priorityDate: '',
    visaCategory: '',
    country: 'india'
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check system preference on mount
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark');
    }
  }, []);

  useEffect(() => {
    // Update document class when theme changes
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

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
        historicalTrend: generateMockTrend(),
        estimatedDate: calculateEstimatedDate(new Date(), data.waitTime)
      });
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

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
    <div className="min-h-screen transition-colors duration-200 dark:bg-gray-900">
      <div className="max-w-6xl mx-auto space-y-6 p-4">
        <Button
          variant="outline"
          size="icon"
          className="fixed top-4 right-4 dark:border-gray-700 dark:hover:bg-gray-800"
          onClick={toggleTheme}
        >
          {theme === 'dark' ? (
            <Sun className="h-4 w-4 text-yellow-500" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>

        <Card className="border-t-4 border-t-green-500 dark:bg-gray-800 dark:border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 dark:text-white">
              <Calendar className="w-6 h-6 text-green-500" />
              Visa Bulletin Priority Date Predictor
            </CardTitle>
            <CardDescription className="dark:text-gray-400">
              Get estimated processing times for employment-based and family-based categories
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2 dark:text-gray-300">
                    <CalendarClock className="w-4 h-4 text-gray-500" />
                    Priority Date
                  </label>
                  <input
                    type="date"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    value={formData.priorityDate}
                    onChange={(e) => setFormData({...formData, priorityDate: e.target.value})}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2 dark:text-gray-300">
                    <Globe className="w-4 h-4 text-gray-500" />
                    Country of Chargeability
                  </label>
                  <Select 
                    defaultValue="india"
                    onValueChange={(value) => setFormData({...formData, country: value})}
                  >
                    <SelectTrigger className="dark:bg-gray-700 dark:border-gray-600 dark:text-white">
                      <SelectValue placeholder="Select country" />
                    </SelectTrigger>
                    <SelectContent className="dark:bg-gray-800">
                      <SelectItem value="india">India</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <Button
                    type="button"
                    variant={activeCategory === 'employment' ? 'default' : 'outline'}
                    className={`flex-1 flex items-center justify-center gap-2 dark:border-gray-600 ${
                      activeCategory === 'employment' ? 'bg-green-500 hover:bg-green-600 dark:bg-green-600 dark:hover:bg-green-700' : 'dark:text-gray-300'
                    }`}
                    onClick={() => setActiveCategory('employment')}
                  >
                    <Briefcase className="w-4 h-4" />
                    Employment Based
                  </Button>
                  <Button
                    type="button"
                    variant={activeCategory === 'family' ? 'default' : 'outline'}
                    className={`flex-1 flex items-center justify-center gap-2 dark:border-gray-600 ${
                      activeCategory === 'family' ? 'bg-green-500 hover:bg-green-600 dark:bg-green-600 dark:hover:bg-green-700' : 'dark:text-gray-300'
                    }`}
                    onClick={() => setActiveCategory('family')}
                  >
                    <Users className="w-4 h-4" />
                    Family Based
                  </Button>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2 dark:text-gray-300">
                    <TrendingUp className="w-4 h-4 text-gray-500" />
                    Visa Category
                  </label>
                  <Select onValueChange={(value) => setFormData({...formData, visaCategory: value})}>
                    <SelectTrigger className="dark:bg-gray-700 dark:border-gray-600 dark:text-white">
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent className="dark:bg-gray-800">
                      {activeCategory === 'employment' ? (
                        <>
                          <SelectItem value="EB1">EB1 - Priority Workers</SelectItem>
                          <SelectItem value="EB2">EB2 - Advanced Degrees</SelectItem>
                          <SelectItem value="EB3">EB3 - Skilled Workers</SelectItem>
                        </>
                      ) : (
                        <>
                          <SelectItem value="F1">F1 - Unmarried Sons/Daughters of USC</SelectItem>
                          <SelectItem value="F2A">F2A - Spouses/Children of LPR</SelectItem>
                          <SelectItem value="F2B">F2B - Unmarried Sons/Daughters of LPR</SelectItem>
                          <SelectItem value="F3">F3 - Married Sons/Daughters of USC</SelectItem>
                          <SelectItem value="F4">F4 - Siblings of Adult USC</SelectItem>
                        </>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button 
                type="submit" 
                className="w-full bg-green-500 hover:bg-green-600 dark:bg-green-600 dark:hover:bg-green-700" 
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
            <Card className="dark:bg-gray-800 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="dark:text-white">Prediction Results</CardTitle>
                <CardDescription className="dark:text-gray-400">Based on historical visa bulletin data analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
                    <h3 className="font-semibold text-lg text-green-700 dark:text-green-400">Estimated Wait Time</h3>
                    <p className="text-4xl font-bold text-green-700 dark:text-green-400">
                      {prediction.waitTime} months
                    </p>
                    <p className="text-sm text-green-600 dark:text-green-400 mt-2">
                      Estimated approval: {prediction.estimatedDate}
                    </p>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm dark:text-gray-300">
                      <span>Confidence Level</span>
                      <span className="font-semibold">{prediction.confidence}%</span>
                    </div>
                    <Progress value={prediction.confidence} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="dark:bg-gray-800 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="dark:text-white">Historical Movement Trend</CardTitle>
                <CardDescription className="dark:text-gray-400">Priority date movement over the last 12 months</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={prediction.historicalTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                      <XAxis 
                        dataKey="date" 
                        tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short' })}
                        stroke={theme === 'dark' ? '#9ca3af' : '#4b5563'}
                      />
                      <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#4b5563'} />
                      <Tooltip 
                        formatter={(value) => [`${value} days`, 'Movement']}
                        labelFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                        contentStyle={{
                          backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
                          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                          color: theme === 'dark' ? '#ffffff' : '#000000'
                        }}
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
    </div>
  );
}