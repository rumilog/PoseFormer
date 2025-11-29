<?php
// =====================================================================
// !!! CRITICAL STEP FOR HOSTING !!!
// 
// 1. Get your Gemini API Key.
// 2. Replace the placeholder below with your actual API key.
// 
// THIS IS THE ONLY FILE WHERE THE KEY IS STORED.
// =====================================================================
$api_key = "Type in your API key here";
$api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=" . $api_key;

// Set content headers for security and JSON response
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *"); // Adjust this for production to your specific domain

// Check if the request method is POST
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405); // Method Not Allowed
    echo json_encode(['error' => 'Only POST requests are allowed.']);
    exit;
}

// Get the raw JSON data sent from the JavaScript client
$json_data = file_get_contents("php://input");

// Initialize cURL for the server-to-server call
$ch = curl_init($api_url);

// Set cURL options
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");
curl_setopt($ch, CURLOPT_POSTFIELDS, $json_data);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, array(
    'Content-Type: application/json',
    'Content-Length: ' . strlen($json_data)
));

// Execute the request and get the response
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

// Handle cURL errors
if (curl_errno($ch)) {
    http_response_code(500);
    echo json_encode(['error' => 'cURL Error: ' . curl_error($ch)]);
} else {
    // Forward the HTTP status code and response body from the Gemini API
    http_response_code($http_code);
    echo $response;
}

curl_close($ch);

?>