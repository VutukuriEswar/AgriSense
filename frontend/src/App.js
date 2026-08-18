import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import '@/App.css';
import 'leaflet/dist/leaflet.css';
import Dashboard from '@/pages/Dashboard';
import DiseaseDetection from '@/pages/DiseaseDetection';
import { Leaf, Bug, BarChart3 } from 'lucide-react';
import { Toaster } from '@/components/ui/sonner';

function App() {
  const [activeNav, setActiveNav] = useState('dashboard');

  return (
    <BrowserRouter>
      <div className="App min-h-screen bg-[#F0FDF4]">
        <Toaster position="top-right" />

        <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 border-b border-green-100">
          <div className="px-6 lg:px-12 py-4 flex items-center justify-between">
            <Link to="/" onClick={() => setActiveNav('dashboard')} className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center shadow-sm">
                <Leaf className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-green-900 tracking-tight" style={{ fontFamily: 'Manrope' }}>
                AgriSense
              </h1>
            </Link>

            <nav className="hidden md:flex items-center gap-2">
              <NavLink to="/" icon={BarChart3} label="Dashboard" active={activeNav === 'dashboard'} onClick={() => setActiveNav('dashboard')} />
              <NavLink to="/disease" icon={Bug} label="Disease AI" active={activeNav === 'disease'} onClick={() => setActiveNav('disease')} />
            </nav>
          </div>
        </header>

        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/disease" element={<DiseaseDetection />} />
          </Routes>
        </main>

        <footer className="mt-20 border-t border-green-100 bg-white py-8 text-center text-sm text-green-800">
          <p>&copy; 2025 AgriSense. Smart Farming Solutions.</p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

function NavLink({ to, icon: Icon, label, active, onClick }) {
  return (
    <Link
      to={to}
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${active ? 'bg-green-600 text-white shadow-sm' : 'text-green-800 hover:bg-green-50'
        }`}
    >
      <Icon className="w-4 h-4" />
      <span className="font-medium text-sm">{label}</span>
    </Link>
  );
}

export default App;