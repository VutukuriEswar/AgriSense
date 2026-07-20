import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Bug, MapPin, Leaf, ShieldCheck, Sprout, BrainCircuit } from 'lucide-react';

const Dashboard = () => {
  return (
    <div className="px-6 lg:px-12 py-12">
      <div className="mb-12 text-center max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-full text-sm font-medium mb-6">
          <Sprout className="w-4 h-4" />
          Smart Farming Technology
        </div>
        <h1 className="text-4xl lg:text-5xl font-bold text-green-900 mb-6 leading-tight">
          Protect Your Crops with <span className="text-green-600">AI Intelligence</span>
        </h1>
        <p className="text-lg text-gray-600 leading-relaxed">
          AgriSense combines satellite weather data with deep learning models to predict crop diseases, identify risks in real-time, and provide actionable insights for farmers. Monitor your farmlands, detect plant diseases instantly, and stay ahead of agricultural threats.
        </p>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold text-green-900 mb-6 text-center">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-white rounded-xl border border-green-100 shadow-sm">
            <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mx-auto mb-4">
              <MapPin className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="font-semibold text-green-900 mb-2">1. Select Location</h3>
            <p className="text-sm text-gray-500">Click on the map to analyze weather conditions and risks for any area.</p>
          </div>
          <div className="text-center p-6 bg-white rounded-xl border border-green-100 shadow-sm">
            <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mx-auto mb-4">
              <BrainCircuit className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="font-semibold text-green-900 mb-2">2. Analyze Data</h3>
            <p className="text-sm text-gray-500">Our AI cross-references weather patterns with disease thresholds.</p>
          </div>
          <div className="text-center p-6 bg-white rounded-xl border border-green-100 shadow-sm">
            <div className="w-12 h-12 bg-red-50 rounded-lg flex items-center justify-center mx-auto mb-4">
              <ShieldCheck className="w-6 h-6 text-red-600" />
            </div>
            <h3 className="font-semibold text-green-900 mb-2">3. Get Precautions</h3>
            <p className="text-sm text-gray-500">Receive specific recommendations to prevent crop loss.</p>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold text-green-900 mb-6 text-center">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
          <Link to="/disease" className="block group">
            <Card className="border-green-100 hover:shadow-lg transition-all h-full hover:border-green-300 cursor-pointer">
              <CardContent className="p-8 flex items-center gap-6">
                <div className="p-4 bg-red-50 rounded-xl group-hover:bg-red-100 transition">
                  <Bug className="w-8 h-8 text-red-600" />
                </div>
                <div>
                  <CardTitle className="text-green-900 mb-1">Disease Detection</CardTitle>
                  <CardDescription>Upload a leaf image to identify diseases instantly using AI.</CardDescription>
                </div>
              </CardContent>
            </Card>
          </Link>

          <Link to="/farmland" className="block group">
            <Card className="border-green-100 hover:shadow-lg transition-all h-full hover:border-green-300 cursor-pointer">
              <CardContent className="p-8 flex items-center gap-6">
                <div className="p-4 bg-blue-50 rounded-xl group-hover:bg-blue-100 transition">
                  <MapPin className="w-8 h-8 text-blue-600" />
                </div>
                <div>
                  <CardTitle className="text-green-900 mb-1">Location Intelligence</CardTitle>
                  <CardDescription>Check weather risks and find nearby farmlands on the map.</CardDescription>
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;