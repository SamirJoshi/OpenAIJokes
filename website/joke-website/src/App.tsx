import React from 'react'
import './App.css'
import JokeGenerator from 'components/JokeGenerator/JokeGenerator'
import Blog from 'components/Blog/Blog'
import DatasetAnalysis from 'components/DatasetAnalysis/DatasetAnalysis'
import References from 'components/References/References'

const App: React.FC = () => {
  return (
    <div className="App"> 
      <JokeGenerator />
      <DatasetAnalysis />
      <Blog />
      <References />
    </div>
  );
}

export default App
