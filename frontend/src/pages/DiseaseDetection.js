import React, { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Upload, Bug, AlertTriangle, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const DiseaseDetection = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      toast.error("Please upload an image first");
      return;
    }

    const formData = new FormData();
    formData.append('file', image);

    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post(`${API}/predict/disease`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);

      if (!res.data.is_valid_plant) {
        toast.warning(res.data.message);
      } else {
        toast.success(`Detected: ${res.data.predicted_class}`);
      }
    } catch (error) {
      toast.error("Analysis failed. Please check console.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const getProbColor = (prob) => {
    if (prob > 0.7) return 'bg-red-500 text-white';
    if (prob > 0.4) return 'bg-orange-400 text-white';
    if (prob > 0.1) return 'bg-yellow-200 text-yellow-800';
    return 'bg-gray-100 text-gray-600';
  };

  return (
    <div className="px-6 lg:px-12 py-12">
      <div className="mb-8 flex items-center gap-4">
        <div className="p-3 bg-red-50 rounded-lg">
          <Bug className="w-6 h-6 text-red-600" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-green-900">Disease AI Detection</h1>
          <p className="text-gray-600">Upload a clear image of a plant leaf to identify potential diseases</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="border-green-100">
          <CardHeader>
            <CardTitle className="text-green-900">Upload Sample</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="border-2 border-dashed border-green-200 rounded-lg p-8 text-center hover:border-green-400 transition cursor-pointer relative min-h-[300px] flex items-center justify-center">
              <input type="file" accept="image/*" onChange={handleImageChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
              {preview ? (
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg object-contain shadow-md" />
              ) : (
                <div className="text-green-600">
                  <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="font-medium">Click or drag image to upload</p>
                  <p className="text-sm text-gray-400 mt-1">JPG, PNG up to 10MB</p>
                </div>
              )}
            </div>

            <Button onClick={analyzeImage} disabled={loading || !image} className="w-full mt-6 bg-green-600 hover:bg-green-700 h-11">
              {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing...</> : <><Bug className="w-4 h-4 mr-2" /> Detect Disease</>}
            </Button>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {result ? (
            result.is_valid_plant ? (
              <Card className="border-green-200 shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-green-900">
                    <span>Top Prediction: {result.predicted_class}</span>
                    <Badge className="bg-green-600 text-white text-base px-3 py-1">{(result.confidence * 100).toFixed(1)}%</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="border-t pt-4">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">All Possibilities:</h4>
                    <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                      {result.all_probabilities.map((item, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 rounded bg-white border hover:shadow-sm transition">
                          <span className="text-sm font-medium text-gray-800">{item.disease}</span>
                          <Badge className={getProbColor(item.probability)}>
                            {(item.probability * 100).toFixed(2)}%
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="border-red-200 bg-red-50 shadow-md">
                <CardContent className="p-6 text-center">
                  <AlertTriangle className="w-16 h-12 text-red-400 mx-auto mb-4" />
                  <h3 className="font-bold text-red-800 text-lg mb-2">Invalid Image</h3>
                  <p className="text-red-600 mb-4">{result.message}</p>
                  <div className="bg-white p-4 rounded-lg text-left text-sm text-gray-600 border border-red-100">
                    <p className="font-semibold mb-2">Tips for best results:</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Ensure the image is focused and clear.</li>
                      <li>Take a close-up of the affected leaf.</li>
                      <li>Avoid shadows and poor lighting.</li>
                      <li>Make sure it is a picture of a plant.</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            )
          ) : (
            <div className="h-full flex items-center justify-center border-2 border-dashed border-gray-200 rounded-lg min-h-[300px]">
              <div className="text-center text-gray-400">
                <Bug className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p>Analysis results will appear here</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DiseaseDetection;