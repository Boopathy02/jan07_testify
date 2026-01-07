import './App.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import 'react-toastify/dist/ReactToastify.css';

import { BrowserRouter, Route, Routes, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';

import Home from './components/home';
import Input from './components/inputs';
import Login from './components/login';
import Signup from './components/signup';
import TestRunner from './components/testrunner';
import Prompts from './components/prompts';



const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/login' element={<Login />} />
        <Route path='/signup' element={<Signup />} />
        
        <Route
          path='/'
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />
        <Route
          path='/editor'
          element={
            <ProtectedRoute>
              <Home editorOnly />
            </ProtectedRoute>
          }
        />
        <Route
          path='/input'
          element={
            <ProtectedRoute>
              <Input />
            </ProtectedRoute>
          }
        />
        <Route
            path='/test-runner'
            element={
              <ProtectedRoute>
                <TestRunner />
              </ProtectedRoute>
            }
          />
          <Route
            path='/prompts'
            element={
              <ProtectedRoute>
                <Prompts />
              </ProtectedRoute>
            }
          />  

      </Routes>
      <ToastContainer position="top-right" autoClose={4000} newestOnTop closeOnClick pauseOnHover draggable theme="colored" />
    </BrowserRouter>
  );
}

export default App;
