import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import { Search, MapPin, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const customIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

const userIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

function LocationMarker({ onLocationSelect }) {
    useMapEvents({
        click(e) {
            onLocationSelect(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
}

function MapController({ center }) {
    const map = useMap();
    useEffect(() => {
        if (center) {
            map.flyTo(center, 13, { duration: 1.5 });
        }
    }, [center, map]);
    return null;
}

const MapComponent = ({ onLocationSelect, markers = [], center = [20.5937, 78.9629] }) => {
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [selectedPosition, setSelectedPosition] = useState(null);
    const [mapCenter, setMapCenter] = useState(center);
    const searchTimeout = useRef(null);

    const handleSearch = async (query) => {
        if (!query || query.length < 3) {
            setSearchResults([]);
            return;
        }

        setIsSearching(true);
        try {
            const res = await axios.get(`${API}/farmland/search?query=${query}`);
            setSearchResults(res.data);
        } catch (error) {
            console.error("Search error:", error);
        } finally {
            setIsSearching(false);
        }
    };

    const handleSearchInputChange = (e) => {
        const query = e.target.value;
        setSearchQuery(query);

        if (searchTimeout.current) {
            clearTimeout(searchTimeout.current);
        }

        searchTimeout.current = setTimeout(() => {
            handleSearch(query);
        }, 500);
    };

    const handleResultClick = (result) => {
        const { lat, lon, name } = result;
        setMapCenter([lat, lon]);
        setSelectedPosition({ lat, lon, name });
        onLocationSelect(lat, lon, name);
        setSearchResults([]);
        setSearchQuery(name);
        toast.success(`Selected: ${name.split(',')[0]}`);
    };

    const handleMapClick = (lat, lng) => {
        setSelectedPosition({ lat, lon: lng, name: 'Selected Point' });
        onLocationSelect(lat, lng);
        setSearchQuery(`${lat.toFixed(4)}, ${lng.toFixed(4)}`);
    };

    const handleMarkerClick = (marker) => {
        setMapCenter([marker.lat, marker.lon]);
        setSelectedPosition({ lat: marker.lat, lon: marker.lon, name: marker.name });
        onLocationSelect(marker.lat, marker.lon, marker.address || marker.name);
        toast.success(`Selected: ${marker.name || 'Farmland'}`);
    };

    return (
        <div className="relative h-full w-full rounded-lg overflow-hidden border border-green-200">
            <div className="absolute top-3 left-3 right-3 z-[1000]">
                <div className="relative">
                    <div className="flex items-center bg-white rounded-lg shadow-lg border border-gray-200">
                        <Search className="w-5 h-5 text-gray-400 ml-3" />
                        <input
                            type="text"
                            placeholder="Search place..."
                            value={searchQuery}
                            onChange={handleSearchInputChange}
                            className="w-full p-3 pl-2 rounded-lg text-sm focus:outline-none"
                        />
                        {isSearching && <Loader2 className="w-4 h-4 animate-spin mr-3 text-gray-400" />}
                    </div>

                    {searchResults.length > 0 && (
                        <div className="search-results-dropdown mt-1">
                            {searchResults.map((result, index) => (
                                <div
                                    key={index}
                                    className="search-result-item"
                                    onClick={() => handleResultClick(result)}
                                >
                                    <div className="flex items-start gap-2">
                                        <MapPin className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                                        <div className="flex-1">
                                            <p className="font-medium text-gray-800">{result.name.split(',')[0]}</p>
                                            <p className="text-xs text-gray-500 truncate">{result.name}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <MapContainer
                center={mapCenter}
                zoom={5}
                style={{ height: '100%', width: '100%' }}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationMarker onLocationSelect={handleMapClick} />
                <MapController center={mapCenter} />

                {selectedPosition && (
                    <Marker position={[selectedPosition.lat, selectedPosition.lon]} icon={userIcon}>
                        <Popup>
                            <div className="text-sm">
                                <strong>Selected Location</strong>
                                <p className="text-xs text-gray-600">{selectedPosition.name || 'Custom Point'}</p>
                            </div>
                        </Popup>
                    </Marker>
                )}

                {markers.map((marker, index) => (
                    <Marker
                        key={marker.id || index}
                        position={[marker.lat, marker.lon]}
                        icon={customIcon}
                        eventHandlers={{
                            click: () => handleMarkerClick(marker)
                        }}
                    >
                        <Popup>
                            <div className="text-sm">
                                <strong className="text-green-800">{marker.name || "Farmland"}</strong>
                                {marker.address && <p className="text-gray-600 mt-1 text-xs">{marker.address}</p>}
                                {!marker.address && (
                                    <p className="text-gray-400 mt-1 text-xs">
                                        {marker.lat.toFixed(4)}, {marker.lon.toFixed(4)}
                                    </p>
                                )}
                                {marker.notes && <p className="text-blue-600 mt-1 text-xs italic">"{marker.notes}"</p>}
                                <button
                                    className="mt-2 text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 transition w-full"
                                    onClick={() => handleMarkerClick(marker)}
                                >
                                    Analyze Location
                                </button>
                            </div>
                        </Popup>
                    </Marker>
                ))}
            </MapContainer>
        </div>
    );
};

export default MapComponent;