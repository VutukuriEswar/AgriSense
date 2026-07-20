import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { MapPin, Plus, List, Search, Loader2, Thermometer, Droplets, Wind, AlertTriangle, CheckCircle, Cloud, Info, Trash2, X, AlertCircle, Camera, Upload, Bug } from 'lucide-react';
import { toast } from 'sonner';
import MapComponent from '@/components/ui/MapComponent';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const FarmlandManager = () => {
    const [farmlands, setFarmlands] = useState([]);
    const [formData, setFormData] = useState({ lat: '', lon: '', name: '', notes: '' });
    const [currentAddress, setCurrentAddress] = useState('');
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisData, setAnalysisData] = useState(null);
    const [editMode, setEditMode] = useState(false);
    const [editId, setEditId] = useState(null);

    const [searchQuery, setSearchQuery] = useState('');
    const [searching, setSearching] = useState(false);

    const [selectedMapLocation, setSelectedMapLocation] = useState(null);

    const [plantImage, setPlantImage] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);

    useEffect(() => {
        fetchFarmlands();
    }, []);

    const fetchFarmlands = async () => {
        try {
            const res = await axios.get(`${API}/farmland/history`);
            setFarmlands(res.data);
        } catch (error) { console.error(error); }
    };

    const handleLocationSelect = async (latitude, longitude, addressFromMap = null) => {
        const lat = parseFloat(latitude);
        const lon = parseFloat(longitude);

        setFormData({ ...formData, lat: lat.toString(), lon: lon.toString() });
        setAnalysisData(null);
        setSelectedMapLocation({ lat, lon });
        setPlantImage(null);
        setImagePreview(null);

        if (addressFromMap) {
            setCurrentAddress(addressFromMap);
        } else {
            setCurrentAddress('Fetching address...');
            try {
                const res = await axios.get(`${API}/farmland/reverse-geocode?lat=${lat}&lon=${lon}`);
                setCurrentAddress(res.data.address);
            } catch (error) { setCurrentAddress('Could not fetch address'); }
        }
    };

    const handleSearch = async () => {
        if (!searchQuery.trim()) return;
        setSearching(true);
        try {
            const res = await axios.get(`${API}/farmland/search?query=${encodeURIComponent(searchQuery)}`);
            if (res.data.length > 0) {
                const first = res.data[0];
                handleLocationSelect(first.lat, first.lon, first.name);
                setSearchQuery(first.name);
            } else { toast.error("No locations found"); }
        } catch (error) { toast.error("Search failed"); }
        finally { setSearching(false); }
    };

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setPlantImage(file);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const handleAnalyze = async () => {
        if (!formData.lat || !formData.lon) {
            toast.error("Please select a location on the map first");
            return;
        }

        setAnalyzing(true);
        setAnalysisData(null);
        try {
            if (plantImage) {
                const multimodalFormData = new FormData();
                multimodalFormData.append('file', plantImage);
                multimodalFormData.append('lat', formData.lat);
                multimodalFormData.append('lon', formData.lon);

                const res = await axios.post(`${API}/farmland/analyze-multimodal`, multimodalFormData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });

                setAnalysisData(res.data);

                if (!res.data.ml_prediction.is_valid_plant) {
                    toast.warning(res.data.ml_prediction.message);
                } else {
                    toast.success("Multimodal analysis complete!");
                }
            } else {
                const res = await axios.get(`${API}/farmland/analyze-location?lat=${formData.lat}&lon=${formData.lon}`);
                setAnalysisData({ ...res.data, ml_prediction: null });
                toast.success("Location analysis complete!");
            }
        } catch (error) {
            toast.error("Failed to analyze location");
            console.error(error);
        } finally {
            setAnalyzing(false);
        }
    };

    const resetForm = () => {
        setFormData({ lat: '', lon: '', name: '', notes: '' });
        setCurrentAddress('');
        setEditMode(false);
        setEditId(null);
        setAnalysisData(null);
        setSelectedMapLocation(null);
        setPlantImage(null);
        setImagePreview(null);
    };

    const handleSave = async (e) => {
        e.preventDefault();
        if (!formData.lat || !formData.lon) { toast.error("Please select a location"); return; }

        setLoading(true);
        try {
            if (editMode && editId) {
                await axios.put(`${API}/farmland/${editId}`, { name: formData.name, notes: formData.notes });
                toast.success("Farmland updated");
            } else {
                await axios.post(`${API}/farmland/save`, formData);
                toast.success("Farmland saved");
            }
            resetForm();
            fetchFarmlands();
        } catch (error) { toast.error("Failed to save"); }
        finally { setLoading(false); }
    };

    const handleCardClick = (farm) => {
        setFormData({
            lat: farm.lat.toString(), lon: farm.lon.toString(),
            name: farm.name, notes: farm.notes || ''
        });
        setCurrentAddress(farm.address || 'Address not available');
        setEditMode(true);
        setEditId(farm.id);
        setAnalysisData(null);
        setSelectedMapLocation({ lat: farm.lat, lon: farm.lon });
        setPlantImage(null);
        setImagePreview(null);
    };

    const handleDelete = async (e, id) => {
        e.stopPropagation();
        if (!window.confirm("Delete this farmland?")) return;
        try {
            await axios.delete(`${API}/farmland/${id}`);
            toast.success("Deleted");
            fetchFarmlands();
        } catch (error) { toast.error("Failed to delete"); }
    };

    const getUrgencyColor = (urgency) => {
        switch (urgency) {
            case 'Critical': return 'bg-red-100 text-red-700 border-red-200';
            case 'High': return 'bg-orange-100 text-orange-700 border-orange-200';
            case 'Medium': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
            default: return 'bg-gray-100 text-gray-700 border-gray-200';
        }
    };

    return (
        <div className="px-6 lg:px-12 py-12">
            <div className="mb-8 flex items-center gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                    <MapPin className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                    <h1 className="text-3xl font-bold text-green-900">Location Intelligence</h1>
                    <p className="text-gray-600">Select a location to analyze environmental risks. Optionally add a plant photo for AI diagnosis.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                <div className="lg:col-span-2 space-y-4">
                    <div className="flex gap-2">
                        <Input placeholder="Search location (e.g., Vijayawada)" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSearch()} />
                        <Button onClick={handleSearch} disabled={searching}>{searching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}</Button>
                    </div>

                    <Card className="border-green-100 h-[500px]">
                        <CardContent className="p-0 h-full relative">
                            <MapComponent onLocationSelect={handleLocationSelect} markers={farmlands} selectedLocation={selectedMapLocation} />
                            {analyzing && (
                                <div className="absolute top-16 left-4 bg-white px-4 py-2 rounded-lg shadow-lg text-sm flex items-center gap-2 z-[1000]">
                                    <Loader2 className="w-4 h-4 animate-spin text-green-600" />
                                    Analyzing...
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>

                <div className="space-y-6">
                    <Card className="border-green-100">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-green-900 text-base">
                                {editMode ? <Search className="w-5 h-5" /> : <Plus className="w-5 h-5" />}
                                {editMode ? 'Edit Farmland' : 'Save New Farmland'}
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <form onSubmit={handleSave} className="space-y-3">
                                <div className="p-3 bg-gray-50 rounded-lg space-y-1">
                                    <div className="text-xs text-gray-500 truncate" title={currentAddress}>
                                        <span className="font-medium">Address:</span> {currentAddress || 'Select location on map'}
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-xs pt-1 border-t border-gray-200 mt-1">
                                        <div><span className="font-medium text-gray-500">Lat:</span> {formData.lat || '--'}</div>
                                        <div><span className="font-medium text-gray-500">Lon:</span> {formData.lon || '--'}</div>
                                    </div>
                                </div>

                                <div>
                                    <Label className="text-xs">Farm Name</Label>
                                    <Input placeholder="My Farm" value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} className="h-9" />
                                </div>
                                <div>
                                    <Label className="text-xs">Notes</Label>
                                    <Input placeholder="Optional notes..." value={formData.notes} onChange={(e) => setFormData({ ...formData, notes: e.target.value })} className="h-9" />
                                </div>

                                <div className="grid grid-cols-2 gap-2">
                                    <Button type="submit" disabled={loading} size="sm" className="bg-green-600 hover:bg-green-700">
                                        {loading ? 'Saving...' : (editMode ? 'Update' : 'Save')}
                                    </Button>
                                    {editMode && (
                                        <Button type="button" variant="outline" size="sm" onClick={resetForm}>
                                            <X className="w-4 h-4 mr-1" /> Cancel
                                        </Button>
                                    )}
                                </div>
                            </form>
                        </CardContent>
                    </Card>

                    {analysisData && (
                        <Card className="border-blue-100 bg-blue-50/50">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="font-semibold text-green-900 flex items-center gap-2">
                                        <Cloud className="w-4 h-4" />
                                        Current Weather
                                    </h3>
                                    <Badge variant="outline" className="bg-white">{analysisData.season}</Badge>
                                </div>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                    <div className="flex items-center gap-2 bg-white/50 p-2 rounded-lg">
                                        <Thermometer className="w-4 h-4 text-red-500" />
                                        <div><p className="text-xs text-gray-500">Temp</p><p className="font-bold text-green-900">{analysisData.weather.temp}°C</p></div>
                                    </div>
                                    <div className="flex items-center gap-2 bg-white/50 p-2 rounded-lg">
                                        <Droplets className="w-4 h-4 text-blue-500" />
                                        <div><p className="text-xs text-gray-500">Humidity</p><p className="font-bold text-green-900">{analysisData.weather.humidity}%</p></div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>

            {analysisData && (
                <div className="space-y-6 mb-8">

                    {analysisData.ml_prediction && (
                        <Card className="border-purple-100">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-purple-800">
                                    <Bug className="w-5 h-5" />
                                    AI Visual Diagnosis
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {analysisData.ml_prediction.is_valid_plant ? (
                                    <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg border border-purple-200">
                                        <div>
                                            <p className="text-sm text-gray-500">Detected Disease</p>
                                            <h3 className="text-xl font-bold text-purple-900">{analysisData.ml_prediction.predicted_class}</h3>
                                        </div>
                                        <Badge className="bg-purple-600 text-white text-base px-3">
                                            {(analysisData.ml_prediction.confidence * 100).toFixed(1)}%
                                        </Badge>
                                    </div>
                                ) : (
                                    <div className="p-4 bg-red-50 rounded-lg border border-red-200 text-center">
                                        <AlertTriangle className="w-6 h-6 text-red-400 mx-auto mb-2" />
                                        <p className="text-red-700 text-sm">{analysisData.ml_prediction.message}</p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}

                    {analysisData.risks.length > 0 && (
                        <Card className="border-red-100">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-red-800">
                                    <AlertTriangle className="w-5 h-5" />
                                    Environmental Risk Predictions
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {analysisData.risks.map((risk, index) => (
                                        <div key={index} className={`p-4 rounded-lg border ${getUrgencyColor(risk.urgency)}`}>
                                            <div className="flex items-start justify-between mb-2">
                                                <div>
                                                    <h4 className="font-bold text-gray-900">{risk.disease}</h4>
                                                    <p className="text-xs font-medium text-gray-600 mt-1">{risk.category}</p>
                                                </div>
                                                <Badge variant="outline" className="bg-white/50 text-xs font-bold">{risk.urgency}</Badge>
                                            </div>

                                            {risk.matched_conditions && risk.matched_conditions.length > 0 && (
                                                <div className="mb-3 p-2 bg-white/40 rounded border border-black/5">
                                                    <p className="text-xs font-semibold text-gray-700 flex items-center gap-1 mb-1">
                                                        <AlertCircle className="w-3 h-3" /> Triggers:
                                                    </p>
                                                    <ul className="text-xs text-gray-600 space-y-0.5">
                                                        {risk.matched_conditions.map((cond, i) => (
                                                            <li key={i} className="flex items-center gap-1">
                                                                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full"></span>
                                                                {cond}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            <div className="space-y-1">
                                                <p className="text-xs font-semibold text-gray-700 flex items-center gap-1">
                                                    <Info className="w-3 h-3" /> Precautions:
                                                </p>
                                                {risk.precautions.slice(0, 3).map((p, i) => (
                                                    <div key={i} className="flex items-start gap-2 text-xs text-gray-700">
                                                        <CheckCircle className="w-3 h-3 mt-0.5 shrink-0" />
                                                        <span>{p}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            )}

            <Card className="border-green-100 mb-8">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-900">
                        <Camera className="w-5 h-5" />
                        Optional: On-Site Plant Diagnosis
                    </CardTitle>
                    <p className="text-sm text-gray-500">Add a photo of a plant from this location for combined AI analysis (Weather + Vision).</p>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-start gap-4">
                        <div className="flex items-center gap-4">
                            <Label htmlFor="plant-image" className="cursor-pointer">
                                <div className="flex items-center gap-2 px-4 py-2 border border-dashed border-green-300 rounded-lg hover:bg-green-50 transition text-green-700">
                                    <Upload className="w-4 h-4" />
                                    <span className="text-sm font-medium">{plantImage ? "Change Image" : "Upload Plant Image"}</span>
                                </div>
                                <Input id="plant-image" type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
                            </Label>
                            {imagePreview && <img src={imagePreview} alt="Preview" className="h-20 w-20 object-cover rounded-md border" />}
                        </div>

                        <Button onClick={handleAnalyze} disabled={analyzing || !formData.lat} className="bg-blue-600 hover:bg-blue-700">
                            {analyzing ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Search className="w-4 h-4 mr-2" />}
                            {analyzing ? "Analyzing..." : "Analyze Location" + (plantImage ? " with Image" : "")}
                        </Button>
                    </div>
                </CardContent>
            </Card>

            <Card className="border-green-100">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-900">
                        <List className="w-5 h-5" />
                        Saved Farmlands
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {farmlands.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {farmlands.map((land) => (
                                <div key={land.id} className="p-4 border border-gray-100 rounded-lg bg-white text-sm hover:border-green-300 cursor-pointer transition relative group" onClick={() => handleCardClick(land)}>
                                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition">
                                        <Button variant="ghost" size="icon" className="h-6 w-6 bg-white shadow-sm border text-gray-400 hover:text-red-600 hover:bg-red-50" onClick={(e) => handleDelete(e, land.id)}>
                                            <Trash2 className="w-3 h-3" />
                                        </Button>
                                    </div>

                                    <h4 className="font-semibold text-green-900 pr-6">{land.name || "Unnamed Farm"}</h4>
                                    <p className="text-xs text-gray-500 mb-1 mt-1">{land.address || 'No address available'}</p>
                                    <p className="text-xs text-gray-400 font-mono">
                                        {parseFloat(land.lat).toFixed(4)}, {parseFloat(land.lon).toFixed(4)}
                                    </p>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-12 text-gray-400">
                            <MapPin className="w-8 h-8 mx-auto mb-2 opacity-50" />
                            <p>No farmlands saved yet</p>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default FarmlandManager;