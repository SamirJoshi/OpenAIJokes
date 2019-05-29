import React from 'react';

const Blog: React.FC = () => {
  return (
    <div className="container blog">
      <div className="container-header">
        <h3>Blog</h3>
      </div>
      <h4>Data</h4>
      stuff that mattered:
      trimming the reddit dataset,
      character trimming,
      title + body

      ^ for reddit


      <h4>Model</h4>
      At the character level, the model has the shape 

      <h4>Hyperparameter Tuning</h4>
      The params to tune are ...


      <h4>Next Steps</h4>
      - transformer Model
      - custom training on top of the OpenAI GPT-2 model

      <h4>Website</h4>
       TS and react with hooks instead of classes bc we wanted to learn TS and hooks.
    </div>
  );
}

export default Blog