import React from 'react'
import Typist from 'react-typist'
import 'App.css'

const JokeGenerator: React.FC = () => {
  return (
    <div className="joke-generator">
      <h1>JOKE GENERATOR</h1>
      <Typist>
        Animate this text.
      </Typist>

      <div className='joke-disclaimer'>
        Disclaimer: These jokes are automatically generated from a dataset ...
        We do not intend for any of these jokes to be harmful and hope that is not the case.
      </div>
    </div>
  );
}

export default JokeGenerator